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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings
import src.models.xformers_compat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/downstream_regression")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_downstream_regression(model_name, task_name, df, seq_col, val_col,
                                     n_folds=5, max_samples=3000):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    
    if model is None:
        return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}
    
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    sequences = df[seq_col].astype(str).tolist()
    values = df[val_col].values.astype(float)
    
    logger.info(f"Extracting embeddings for {model_name} on {task_name} ({len(sequences)} samples)...")
    t0 = time.time()
    
    try:
        embeddings = extract_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=16)
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        CodonModelLoader.release(model, DEVICE)
        return {"model": model_name, "task": task_name, "success": False, "error": str(e)}
    
    embed_time = time.time() - t0
    logger.info(f"Embeddings: shape={embeddings.shape}, time={embed_time:.1f}s")
    CodonModelLoader.release(model, DEVICE)
    
    X = StandardScaler().fit_transform(embeddings)
    y = values
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = []
    fold_spearman = []
    fold_pearson = []
    
    for train_idx, test_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_true = y[test_idx]
        
        r2 = r2_score(y_true, y_pred)
        sr = stats.spearmanr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 else 0
        pr = stats.pearsonr(y_true, y_pred)[0] if len(np.unique(y_true)) > 1 else 0
        
        fold_r2.append(r2)
        fold_spearman.append(sr)
        fold_pearson.append(pr)
    
    result = {
        "model": model_name, "task": task_name,
        "architecture": config.architecture, "params_M": config.params_M,
        "success": True, "n_samples": len(X), "n_folds": n_folds,
        "embedding_dim": int(embeddings.shape[1]),
        "embed_time_s": round(embed_time, 1),
        "R2": round(float(np.mean(fold_r2)), 4),
        "R2_std": round(float(np.std(fold_r2)), 4),
        "Spearman_r": round(float(np.mean(fold_spearman)), 4),
        "Spearman_r_std": round(float(np.std(fold_spearman)), 4),
        "Pearson_r": round(float(np.mean(fold_pearson)), 4),
        "Pearson_r_std": round(float(np.std(fold_pearson)), 4),
        "data_source": "demo" if max_samples and len(df) <= max_samples else "full",
    }
    
    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"  {model_name}/{task_name}: R2={result['R2']:.4f}, Spearman={result['Spearman_r']:.4f}")
    return result


def main():
    from src.data.download import prepare_task4, prepare_task5, prepare_task6
    
    tasks = [
        ("task4_translation_efficiency", prepare_task4(DATA_DIR), "codon_seq", "te_value"),
        ("task5_protein_expression", prepare_task5(DATA_DIR), "codon_seq", "expression"),
        ("task6_mrna_stability", prepare_task6(DATA_DIR), "codon_seq", "halflife"),
    ]
    
    models = ["encodon-80m", "codonbert"]
    
    all_results = []
    for model_name in models:
        for task_name, df, seq_col, val_col in tasks:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} on {task_name}")
            print(f"{'='*60}")
            try:
                result = evaluate_downstream_regression(
                    model_name, task_name, df, seq_col, val_col, max_samples=2000
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "downstream_regression_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Downstream regression complete: {len(all_results)} model-task pairs")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:35s} | R2={r['R2']:.4f} | Spearman={r['Spearman_r']:.4f}")
        else:
            print(f"  {r['model']:20s} | {r['task']:35s} | FAILED: {r.get('error','unknown')[:60]}")


if __name__ == "__main__":
    main()