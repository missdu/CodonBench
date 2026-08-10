import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import time
import logging
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings
import src.models.xformers_compat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/downstream_classification")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dna_to_rna(seq):
    return seq.replace("T", "U")


def evaluate_downstream_classification(model_name, task_name, df, use_rna=False, 
                                         n_folds=5, max_samples=2000):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    if max_samples and len(df) > max_samples:
        # Balance classes
        n_per_class = max_samples // 2
        df_pos = df[df["label"] == 1].sample(n=min(n_per_class, int(df["label"].sum())), random_state=42)
        df_neg = df[df["label"] == 0].sample(n=min(n_per_class, int(len(df) - df["label"].sum())), random_state=42)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)
    
    # Get wt and mut sequences
    wt_seqs = df["wt_codon_seq"].astype(str).tolist()
    mut_seqs = df["mut_codon_seq"].astype(str).tolist()
    labels = df["label"].values.astype(int)
    
    if use_rna:
        wt_seqs = [dna_to_rna(s) for s in wt_seqs]
        mut_seqs = [dna_to_rna(s) for s in mut_seqs]
    
    # Extract embeddings for wt and mut sequences
    logger.info(f"Extracting embeddings for {model_name} on {task_name} ({len(wt_seqs)} samples)...")
    t0 = time.time()
    
    try:
        wt_embs = extract_embeddings(model, tokenizer, wt_seqs, device=DEVICE, batch_size=16)
        mut_embs = extract_embeddings(model, tokenizer, mut_seqs, device=DEVICE, batch_size=16)
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        CodonModelLoader.release(model, DEVICE)
        return {"model": model_name, "task": task_name, "success": False, "error": str(e)}
    
    embed_time = time.time() - t0
    logger.info(f"Embeddings: wt={wt_embs.shape}, mut={mut_embs.shape}, time={embed_time:.1f}s")
    
    CodonModelLoader.release(model, DEVICE)
    
    # Feature: difference embedding (mut - wt)
    X = mut_embs - wt_embs
    
    # Also try concatenation [wt; mut; mut-wt]
    X_concat = np.concatenate([wt_embs, mut_embs, X], axis=1)
    
    y = labels
    
    results_features = {}
    
    for feat_name, X_feat in [("diff", X), ("concat", X_concat)]:
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat)
        
        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_aucs = []
        fold_aups = []
        fold_f1s = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X_scaled[train_idx], y[train_idx])
            y_prob = clf.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            y_true = y[test_idx]
            
            try:
                auc = roc_auc_score(y_true, y_prob)
                aup = average_precision_score(y_true, y_prob)
            except:
                auc = float("nan")
                aup = float("nan")
            f1 = f1_score(y_true, y_pred, average="binary")
            
            fold_aucs.append(auc)
            fold_aups.append(aup)
            fold_f1s.append(f1)
        
        results_features[feat_name] = {
            "ROC-AUC": round(float(np.mean(fold_aucs)), 4),
            "ROC-AUC_std": round(float(np.std(fold_aucs)), 4),
            "PR-AUC": round(float(np.mean(fold_aups)), 4),
            "F1": round(float(np.mean(fold_f1s)), 4),
        }
    
    # Use best feature type
    best_feat = max(results_features.keys(), key=lambda k: results_features[k]["ROC-AUC"])
    
    result = {
        "model": model_name, "task": task_name,
        "architecture": config.architecture, "params_M": config.params_M,
        "success": True, "n_samples": len(X), "n_folds": n_folds,
        "embedding_dim": int(wt_embs.shape[1]),
        "embed_time_s": round(embed_time, 1),
        "best_feature": best_feat,
        **{f"{best_feat}_{k}": v for k, v in results_features[best_feat].items()},
        "all_features": results_features,
    }
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"  {model_name}/{task_name}: ROC-AUC={results_features[best_feat]['ROC-AUC']:.4f} (feat={best_feat})")
    return result


def main():
    from src.data.download import prepare_task2, prepare_task3
    
    # Load task data
    df_task2 = prepare_task2(DATA_DIR)
    df_task3 = prepare_task3(DATA_DIR)
    
    print(f"Task2: {len(df_task2)} samples ({int(df_task2['label'].sum())} patho)")
    print(f"Task3: {len(df_task3)} samples ({int(df_task3['label'].sum())} patho)")
    
    # Models to evaluate
    models_config = [
        ("encodon-80m", False),
        ("codonbert", False),
        ("codonbert_hf", True),
        ("cdsbert", False),
    ]
    
    tasks = [
        ("task2_clinvar_missense", "df_task2"),
        ("task3_synonymous", "df_task3"),
    ]
    
    all_results = []
    
    for model_name, use_rna in models_config:
        for task_name, df_var in tasks:
            df = df_task2 if df_var == "df_task2" else df_task3
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {task_name} (rna={use_rna})")
            print(f"{'='*60}")
            try:
                result = evaluate_downstream_classification(
                    model_name, task_name, df, use_rna=use_rna, max_samples=5000
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "downstream_classification_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Downstream classification complete: {len(all_results)} model-task pairs")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            feat = r.get("best_feature", "diff")
            auc_key = f"{feat}_ROC-AUC"
            print(f"  {r['model']:20s} | {r['task']:30s} | ROC-AUC={r.get(auc_key, 'N/A')} | feat={feat}")
        else:
            print(f"  {r['model']:20s} | {r['task']:30s} | FAILED: {r.get('error','unknown')[:60]}")


if __name__ == "__main__":
    main()