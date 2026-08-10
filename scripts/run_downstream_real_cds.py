import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/downstream_real_cds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16

def dna_to_rna(seq):
    return seq.replace("T", "U")

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

def build_cds_sequence(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)

def evaluate_downstream_cds(model_name, task_name, variants_df, cds_cache,
                             use_rna=False, max_samples=3000):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)
    
    sequences = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        
        seq_str = build_cds_sequence(cds_seq, cpos)
        if seq_str is None:
            skipped += 1
            continue
        
        if use_rna:
            seq_str = dna_to_rna(seq_str)
        
        sequences.append(seq_str)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(sequences)} sequences (skipped {skipped}), P={int(labels.sum())} B={int(len(labels)-labels.sum())}")
    
    if len(sequences) < 100:
        CodonModelLoader.release(model, DEVICE)
        return {"model": model_name, "task": task_name, "success": False, "error": f"Too few: {len(sequences)}"}
    
    # Extract embeddings
    t0 = time.time()
    embeddings = extract_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=16, show_progress=True)
    emb_time = time.time() - t0
    print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")
    
    CodonModelLoader.release(model, DEVICE)
    
    # Classification with 5-fold CV
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    t0 = time.time()
    aucs = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")
    cv_time = time.time() - t0
    
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    
    # Statistical test
    t_stat, p_val = stats.ttest_1samp(aucs, 0.5)
    neg_log10_p = -np.log10(p_val) if p_val > 0 else float("inf")
    
    result = {
        "model": model_name, "task": task_name,
        "architecture": config.architecture, "params_M": config.params_M,
        "success": True, "n_samples": len(sequences),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC_mean": round(float(mean_auc), 4),
        "ROC-AUC_std": round(float(std_auc), 4),
        "ROC-AUC_folds": [round(float(a), 4) for a in aucs],
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "emb_time_s": round(emb_time, 1), "cv_time_s": round(cv_time, 1),
        "use_rna": use_rna, "context_codons": CODON_CONTEXT,
    }
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result: ROC-AUC={mean_auc:.4f}±{std_auc:.4f}, -log10(p)={result['neg_log10_p']}")
    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")
    
    print("Loading ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False, nrows=1000000)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    
    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw): return 1
        if any(k in cs for k in benign_kw): return 0
        return -1
    
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    labeled = snv[snv["label"] >= 0].copy()
    valid = labeled[
        (labeled["ReferenceAlleleVCF"] != "na") &
        (labeled["AlternateAlleleVCF"] != "na") &
        (labeled["ReferenceAlleleVCF"].str.len() == 1) &
        (labeled["AlternateAlleleVCF"].str.len() == 1)
    ].copy()
    
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    valid["cref"] = parsed.apply(lambda x: x[2])
    valid["calt"] = parsed.apply(lambda x: x[3])
    
    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()
    
    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))
    
    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()
    print(f"  Task2 (missense): {len(task2_variants)} (P={(task2_variants['label']==1).sum()}, B={(task2_variants['label']==0).sum()})")
    print(f"  Task3 (synonymous): {len(task3_variants)} (P={(task3_variants['label']==1).sum()}, B={(task3_variants['label']==0).sum()})")
    
    models_config = [
        ("encodon-80m", False),
        ("codonbert", False),
        ("codonbert_hf", True),
    ]
    
    all_results = []
    tasks = [
        ("task2_missense_cds", task2_variants, 3000),
        ("task3_synonymous_cds", task3_variants, 2000),
    ]
    
    for model_name, use_rna in models_config:
        for task_name, task_df, max_samples in tasks:
            if len(task_df) < 100:
                continue
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {task_name} (downstream, rna={use_rna})")
            print(f"{'='*60}")
            try:
                result = evaluate_downstream_cds(
                    model_name, task_name, task_df, cds_cache,
                    use_rna=use_rna, max_samples=max_samples
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "downstream_cds_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Downstream real-CDS evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:25s} | AUC={r['ROC-AUC_mean']:.4f}±{r['ROC-AUC_std']:.4f} | -log10p={r.get('neg_log10_p','N/A')}")
        else:
            print(f"  {r['model']:20s} | {r.get('task','?'):25s} | FAILED: {str(r.get('error','unknown'))[:80]}")

if __name__ == "__main__":
    main()