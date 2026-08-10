import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

logger = logging.getLogger(__name__)


def evaluate_downstream_single(
    model_name: str,
    task_name: str,
    task_data: pd.DataFrame,
    device: str = "cuda:0",
    n_folds: int = 5,
    n_estimators: int = 100,
    random_seed: int = 42,
    embedding_layer: int = -2,
    batch_size: int = 8,
) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, load_meta = CodonModelLoader.load(model_name, device=device)

    if model is None:
        return {
            "model": model_name, "task": task_name,
            "success": False, "error": load_meta.get("error", "load failed"),
            "load_meta": load_meta,
        }

    seq_col = "codon_seq"
    if seq_col not in task_data.columns:
        seq_col = task_data.columns[0]
        logger.warning(f"Using first column '{seq_col}' as sequence column")

    val_col = [c for c in task_data.columns if c != seq_col][0] if len(task_data.columns) > 1 else None
    if val_col is None:
        CodonModelLoader.release(model, device)
        return {"model": model_name, "task": task_name, "success": False, "error": "No value column"}

    sequences = task_data[seq_col].astype(str).tolist()
    labels = task_data[val_col].values.astype(float)

    logger.info(f"Extracting embeddings for {model_name} on {task_name} ({len(sequences)} samples)...")
    t0 = time.time()

    try:
        embeddings = extract_embeddings(
            model, tokenizer, sequences, device=device,
            layer=embedding_layer, batch_size=batch_size,
        )
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        CodonModelLoader.release(model, device)
        return {"model": model_name, "task": task_name, "success": False, "error": str(e)}

    embed_time = time.time() - t0
    logger.info(f"Embeddings: shape={embeddings.shape}, time={embed_time:.1f}s")

    CodonModelLoader.release(model, device)

    X = embeddings
    y = labels

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_metrics = {"R2": [], "pearson_r": [], "spearman_r": []}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_true = y[test_idx]

        r2 = r2_score(y_true, y_pred)
        pr = stats.pearsonr(y_true, y_pred)[0]
        sr = stats.spearmanr(y_true, y_pred)[0]

        fold_metrics["R2"].append(r2)
        fold_metrics["pearson_r"].append(pr)
        fold_metrics["spearman_r"].append(sr)

    avg_metrics = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
    std_metrics = {k + "_std": round(float(np.std(v)), 4) for k, v in fold_metrics.items()}

    result = {
        "model": model_name,
        "task": task_name,
        "architecture": config.architecture,
        "params_M": config.params_M,
        "success": True,
        "n_samples": len(X),
        "n_folds": n_folds,
        "embedding_dim": int(X.shape[1]),
        "embed_time_s": round(embed_time, 1),
        "load_meta": load_meta,
        **avg_metrics,
        **std_metrics,
        "fold_details": {k: [round(float(v), 4) for v in vs] for k, vs in fold_metrics.items()},
    }

    return result


def run_downstream_eval(
    model_names: List[str],
    task_names: List[str],
    data_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    n_folds: int = 5,
) -> List[Dict]:
    from src.data.download import prepare_all_tasks

    all_task_data = prepare_all_tasks(data_dir)

    task_map = {
        "task4_translation_efficiency": "task4_translation_efficiency",
        "task5_protein_expression": "task5_protein_expression",
        "task6_mrna_stability": "task6_mrna_stability",
    }

    results = []
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        for task_name in task_names:
            data_key = task_map.get(task_name)
            if not data_key or data_key not in all_task_data or all_task_data[data_key] is None:
                logger.warning(f"Skip {model_name}/{task_name}: data not available")
                continue

            logger.info(f"Evaluating {model_name} on {task_name}...")
            result = evaluate_downstream_single(
                model_name, task_name, all_task_data[data_key],
                device=device, n_folds=n_folds,
            )
            results.append(result)

            result_file = out_path / f"{model_name}_{task_name}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"  Result: R2={result.get('R2', 'N/A')}, saved to {result_file}")

    summary_file = out_path / "downstream_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"All downstream results saved to {summary_file}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--tasks", default="task4_translation_efficiency,task5_protein_expression", help="Tasks")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        all_cfgs = CodonModelLoader.load_configs()
        model_names = [n for n, c in all_cfgs.items() if c.priority <= 2]

    task_names = [t.strip() for t in args.tasks.split(",")]
    results = run_downstream_eval(model_names, task_names, args.data, args.output, args.device, args.n_folds)

    print(f"\n{'='*60}")
    print(f"Downstream evaluation complete: {len(results)} model-task pairs")
    print(f"{'='*60}")
    for r in results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:35s} | R2={r['R2']:.4f} | Pearson={r['pearson_r']:.4f} | Spearman={r['spearman_r']:.4f}")
        else:
            print(f"  {r['model']:20s} | {r['task']:35s} | FAILED: {r.get('error','unknown')}")
