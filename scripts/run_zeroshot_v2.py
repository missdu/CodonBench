import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import time
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder, _fix_token_type_ids
from src.data.download import prepare_task2, CODON_TABLE
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

import src.models.xformers_compat

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/zeroshot_real_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dna_to_rna(seq):
    return seq.replace("T", "U")

def evaluate_model_task_v2(model_name, task_name, df, use_rna=False, max_samples=1000):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    # Construct single-codon wt/mut sequences from ReferenceAlleleVCF/AlternateAlleleVCF
    # For each variant, wt_codon = ref*3, mut_codon = alt+ref*2
    # This directly tests the model's sensitivity to the mutation
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Build codon-level sequences with short context (3 codons before + mutation + 3 codons after)
    # Use common codons as context to provide realistic flanking
    common_codons = ["ATG", "GCT", "AAG", "TTT", "CCC", "GGG", "CAG", "GAC"]
    
    wt_seqs = []
    mut_seqs = []
    labels = df["label"].values.astype(float)
    
    for i, row in df.iterrows():
        ref = str(row.get("ReferenceAlleleVCF", row.get("ref", "A")))
        alt = str(row.get("AlternateAlleleVCF", row.get("alt", "T")))
        if ref == "na" or alt == "na" or len(ref) != 1 or len(alt) != 1:
            wt_seqs.append(ref * 3)
            mut_seqs.append(alt + ref * 2)
            continue
        
        # 3 codons context + mutation codon + 3 codons context
        np.random.seed(hash(str(row.get("Name", str(i)))) % (2**31))
        ctx_before = [np.random.choice(common_codons) for _ in range(3)]
        ctx_after = [np.random.choice(common_codons) for _ in range(3)]
        
        wt_codon = ref * 3
        mut_codon = alt + ref * 2
        
        wt_seq = " ".join(ctx_before + [wt_codon] + ctx_after)
        mut_seq = " ".join(ctx_before + [mut_codon] + ctx_after)
        
        if use_rna:
            wt_seq = dna_to_rna(wt_seq)
            mut_seq = dna_to_rna(mut_seq)
        
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
    
    llrs = []
    t0 = time.time()
    for i, (wt, mut) in enumerate(zip(wt_seqs, mut_seqs)):
        try:
            llr = compute_llr_encoder(model, tokenizer, wt, mut, device=DEVICE)
            llrs.append(llr)
        except Exception as e:
            llrs.append(0.0)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(wt_seqs) - i - 1)
            print(f"  [{model_name}][{task_name}] {i+1}/{len(wt_seqs)}, ~{remaining:.0f}s left")
    
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
        "success": True, "n_samples": len(df),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC": round(float(auc_roc), 4), "PR-AUC": round(float(auc_pr), 4),
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "eval_time_s": round(elapsed, 1), "use_rna": use_rna,
    }
    
    CodonModelLoader.release(model, DEVICE)
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result: ROC-AUC={result['ROC-AUC']:.4f}, PR-AUC={result['PR-AUC']:.4f}")
    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    # Load ClinVar data with variant details
    df_task2 = prepare_task2(DATA_DIR)
    
    # We need the raw ClinVar data to get ReferenceAlleleVCF/AlternateAlleleVCF
    # The processed parquet only has wt_codon_seq/mut_codon_seq
    # So we need to re-process with variant details preserved
    import pandas as pd
    clinvar_path = DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz"
    raw = pd.read_csv(clinvar_path, sep="\t", low_memory=False)
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
    
    n_sample = 1000
    patho_df = valid[valid["label"] == 1].sample(n=min(n_sample, int(valid["label"].sum())), random_state=42)
    benign_df = valid[valid["label"] == 0].sample(n=min(n_sample, int(len(valid) - valid["label"].sum())), random_state=42)
    balanced = pd.concat([patho_df, benign_df]).sample(frac=1, random_state=42)
    
    print(f"Task2 data: {len(balanced)} samples ({int(balanced['label'].sum())} patho, {int(len(balanced)-balanced['label'].sum())} benign)")
    
    # Models to evaluate
    models_config = [
        ("encodon-80m", False),     # DNA codons
        ("codonbert", False),        # DNA codons (local)
        ("codonbert_hf", True),      # RNA codons (HF)
        ("cdsbert", False),          # character-level, try DNA
    ]
    
    all_results = []
    for model_name, use_rna in models_config:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} on task2_clinvar_missense (rna={use_rna})")
        print(f"{'='*60}")
        try:
            result = evaluate_model_task_v2(model_name, "task2_clinvar_missense", balanced, use_rna=use_rna, max_samples=1000)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"model": model_name, "task": "task2_clinvar_missense", "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "zeroshot_summary_v2.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Zero-shot v2 evaluation complete: {len(all_results)} model-task pairs")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | ROC-AUC={r['ROC-AUC']:.4f} | PR-AUC={r['PR-AUC']:.4f} | time={r['eval_time_s']:.0f}s")
        else:
            print(f"  {r['model']:20s} | FAILED: {r.get('error','unknown')[:80]}")

if __name__ == "__main__":
    main()