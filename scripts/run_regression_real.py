import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import time
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from scipy import stats

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DEVICE = "cuda:2"
DATA_DIR = Path("./data/regression")
OUT_DIR = Path("./results/regression_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def rna_to_dna(seq):
    return seq.replace("U", "T")

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def seq_to_codon_str(seq, use_rna=False):
    if use_rna:
        return " ".join(dna_to_codons(seq))
    else:
        dna = rna_to_dna(seq) if "U" in seq else seq
        return " ".join(dna_to_codons(dna))

def load_regression_data(filename, max_samples=None):
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    
    seq_col = "Sequence" if "Sequence" in df.columns else "CDS"
    val_col = "Value"
    
    df = df[[seq_col, val_col]].dropna()
    df = df.rename(columns={seq_col: "sequence", val_col: "value"})
    
    df = df[df["sequence"].str.len() >= 9]
    df = df[df["sequence"].str.len() % 3 == 0]
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df.reset_index(drop=True)

def evaluate_regression(model_name, task_name, df, use_rna=False):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    sequences = []
    for seq in df["sequence"]:
        try:
            codon_str = seq_to_codon_str(seq, use_rna=use_rna)
            sequences.append(codon_str)
        except:
            sequences.append(None)
    
    valid_mask = [s is not None for s in sequences]
    sequences = [s for s in sequences if s is not None]
    values = df["value"].values[valid_mask]
    
    print(f"  Valid sequences: {len(sequences)}/{len(df)}")
    
    if len(sequences) < 50:
        CodonModelLoader.release(model, DEVICE)
        return {"model": model_name, "task": task_name, "success": False, "error": f"Too few: {len(sequences)}"}
    
    t0 = time.time()
    embeddings = extract_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=16, show_progress=True)
    emb_time = time.time() - t0
    print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")
    
    CodonModelLoader.release(model, DEVICE)
    
    # Regression with Ridge (fast) and RF
    results = {}
    for clf_name, clf in [("Ridge", Ridge(alpha=1.0)), ("RF", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        t0 = time.time()
        r2_scores = cross_val_score(clf, embeddings, values, cv=cv, scoring="r2")
        spearman_scores = []
        for train_idx, test_idx in cv.split(embeddings):
            clf.fit(embeddings[train_idx], values[train_idx])
            pred = clf.predict(embeddings[test_idx])
            r, _ = stats.spearmanr(pred, values[test_idx])
            spearman_scores.append(r)
        cv_time = time.time() - t0
        
        results[clf_name] = {
            "r2_mean": round(float(np.mean(r2_scores)), 4),
            "r2_std": round(float(np.std(r2_scores)), 4),
            "spearman_mean": round(float(np.mean(spearman_scores)), 4),
            "spearman_std": round(float(np.std(spearman_scores)), 4),
            "cv_time_s": round(cv_time, 1),
        }
        print(f"  {clf_name}: R2={np.mean(r2_scores):.4f}+/-{np.std(r2_scores):.4f}, Spearman={np.mean(spearman_scores):.4f}")
    
    result = {
        "model": model_name, "task": task_name,
        "architecture": config.architecture, "params_M": config.params_M,
        "success": True, "n_samples": len(sequences),
        "value_mean": round(float(np.mean(values)), 4),
        "value_std": round(float(np.std(values)), 4),
        "emb_time_s": round(emb_time, 1),
        "use_rna": use_rna,
        **{f"{k}_{mk}": v for k, d in results.items() for mk, v in d.items()},
    }
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    tasks = [
        ("task4_mrfp_expression", "mRFP_Expression.csv", 1459),
        ("task5_ecoli_proteins", "E.Coli_proteins.csv", 3000),
        ("task6_mrna_stability", "mRNA_Stability.csv", 5000),
    ]
    
    models_config = [
        ("encodon-80m", False),
        ("codonbert", False),
        ("codonbert_hf", True),
    ]
    
    all_results = []
    
    for task_name, filename, max_samples in tasks:
        print(f"\nLoading {filename}...")
        df = load_regression_data(filename, max_samples=max_samples)
        print(f"  {len(df)} samples, value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")
        
        for model_name, use_rna in models_config:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {task_name} (rna={use_rna}, n={len(df)})")
            print(f"{'='*60}")
            try:
                result = evaluate_regression(model_name, task_name, df, use_rna=use_rna)
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "regression_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Regression evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            ridge_r2 = r.get("Ridge_r2_mean", "N/A")
            ridge_sp = r.get("Ridge_spearman_mean", "N/A")
            rf_r2 = r.get("RF_r2_mean", "N/A")
            rf_sp = r.get("RF_spearman_mean", "N/A")
            print(f"  {r['model']:20s} | {r['task']:30s} | Ridge: R2={ridge_r2}, Sp={ridge_sp} | RF: R2={rf_r2}, Sp={rf_sp}")
        else:
            print(f"  {r['model']:20s} | {r.get('task','?'):30s} | FAILED: {str(r.get('error','unknown'))[:80]}")

if __name__ == "__main__":
    main()