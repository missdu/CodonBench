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
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder, _fix_token_type_ids

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/zeroshot_real_cds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16  # 16 codons = 48 nt on each side of mutation

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

def build_cds_sequences(cds_seq, cpos, cref, calt, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None, None
    if cds_seq[cpos - 1] != cref:
        return None, None
    
    # Build wt sequence: full CDS with mutation site
    wt_seq = cds_seq
    mut_seq = cds_seq[:cpos-1] + calt + cds_seq[cpos:]
    
    # Extract context window around mutation (in codons)
    # Find which codon the mutation is in
    codon_idx = (cpos - 1) // 3  # 0-indexed codon position
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(wt_seq) // 3, codon_idx + context_codons + 1)
    
    # Extract codon-level subsequences
    wt_codons = dna_to_codons(wt_seq[start_codon*3:end_codon*3])
    mut_codons = dna_to_codons(mut_seq[start_codon*3:end_codon*3])
    
    if not wt_codons or not mut_codons:
        return None, None
    
    # Format as space-separated codons (for codon-level tokenizers)
    wt_str = " ".join(wt_codons)
    mut_str = " ".join(mut_codons)
    
    return wt_str, mut_str

def evaluate_model_zeroshot_cds(model_name, task_name, variants_df, cds_cache, 
                                  use_rna=False, max_samples=1000):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        # Balance pathogenic/benign
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)
    
    wt_seqs = []
    mut_seqs = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cref = row["cref"]
        calt = row["calt"]
        
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        
        wt_str, mut_str = build_cds_sequences(cds_seq, cpos, cref, calt)
        if wt_str is None:
            skipped += 1
            continue
        
        if use_rna:
            wt_str = dna_to_rna(wt_str)
            mut_str = dna_to_rna(mut_str)
        
        wt_seqs.append(wt_str)
        mut_seqs.append(mut_str)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=float)
    print(f"  Built {len(wt_seqs)} sequence pairs (skipped {skipped}), P={int(labels.sum())} B={int(len(labels)-labels.sum())}")
    
    if len(wt_seqs) < 50:
        CodonModelLoader.release(model, DEVICE)
        return {"model": model_name, "task": task_name, "success": False, "error": f"Too few valid pairs: {len(wt_seqs)}"}
    
    llrs = []
    t0 = time.time()
    for i, (wt, mut) in enumerate(zip(wt_seqs, mut_seqs)):
        try:
            llr = compute_llr_encoder(model, tokenizer, wt, mut, device=DEVICE)
            llrs.append(llr)
        except Exception as e:
            llrs.append(0.0)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(wt_seqs) - i - 1)
            print(f"  [{model_name}] {i+1}/{len(wt_seqs)}, ~{remaining:.0f}s left, avg_llr={np.mean(llrs[-100:]):.4f}")
    
    llrs = np.array(llrs)
    elapsed = time.time() - t0
    neg_llrs = -llrs
    
    try:
        auc_roc = roc_auc_score(labels, neg_llrs)
        auc_pr = average_precision_score(labels, neg_llrs)
    except:
        auc_roc = float("nan")
        auc_pr = float("nan")
    
    try:
        patho = neg_llrs[labels == 1]
        benign = neg_llrs[labels == 0]
        if len(patho) > 0 and len(benign) > 0:
            u_stat, p_val = stats.mannwhitneyu(patho, benign, alternative="greater")
            neg_log10_p = -np.log10(p_val) if p_val > 0 else float("inf")
        else:
            u_stat, p_val, neg_log10_p = float("nan"), float("nan"), float("nan")
    except:
        u_stat, p_val, neg_log10_p = float("nan"), float("nan"), float("nan")
    
    result = {
        "model": model_name, "task": task_name,
        "architecture": config.architecture, "params_M": config.params_M,
        "success": True, "n_samples": len(wt_seqs),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC": round(float(auc_roc), 4), "PR-AUC": round(float(auc_pr), 4),
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "mean_llr_patho": round(float(llrs[labels==1].mean()), 4) if len(llrs[labels==1]) > 0 else None,
        "mean_llr_benign": round(float(llrs[labels==0].mean()), 4) if len(llrs[labels==0]) > 0 else None,
        "eval_time_s": round(elapsed, 1), "use_rna": use_rna,
        "context_codons": CODON_CONTEXT,
    }
    
    CodonModelLoader.release(model, DEVICE)
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result: ROC-AUC={result['ROC-AUC']:.4f}, PR-AUC={result['PR-AUC']:.4f}, -log10(p)={result['neg_log10_p']}")
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
    print(f"  Usable variants: {len(variants)} (P={(variants['label']==1).sum()}, B={(variants['label']==0).sum()})")
    
    # Task2: missense (non-synonymous) - p.XxxYyy where Xxx!=Yyy
    # Task3: synonymous - p.Xnnn= format
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
        ("task2_missense_cds", task2_variants, 2000),
        ("task3_synonymous_cds", task3_variants, 2000),
    ]
    
    for model_name, use_rna in models_config:
        for task_name, task_df, max_samples in tasks:
            if len(task_df) < 20:
                print(f"  SKIP {model_name}/{task_name}: too few samples ({len(task_df)})")
                continue
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {task_name} (rna={use_rna}, n={len(task_df)})")
            print(f"{'='*60}")
            try:
                result = evaluate_model_zeroshot_cds(
                    model_name, task_name, task_df, cds_cache,
                    use_rna=use_rna, max_samples=max_samples
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "zeroshot_cds_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Zero-shot real-CDS evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:25s} | ROC-AUC={r['ROC-AUC']:.4f} | PR-AUC={r['PR-AUC']:.4f} | -log10p={r.get('neg_log10_p','N/A')} | time={r['eval_time_s']:.0f}s")
        else:
            print(f"  {r['model']:20s} | {r.get('task','?'):25s} | FAILED: {str(r.get('error','unknown'))[:80]}")

if __name__ == "__main__":
    main()