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
from src.eval.evaluation_utils import compute_llr_encoder, compute_llr_decoder

logger = logging.getLogger(__name__)


def evaluate_zeroshot_single(
    model_name: str,
    task_name: str,
    task_data: pd.DataFrame,
    device: str = "cuda:0",
    max_samples: int = None,
) -> Dict[str, Any]:
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, load_meta = CodonModelLoader.load(model_name, device=device)

    if model is None:
        logger.error(f"Cannot load {model_name}: {load_meta['error']}")
        return {
            "model": model_name, "task": task_name,
            "success": False, "error": load_meta["error"],
            "load_meta": load_meta,
        }

    wt_col = "wt_codon_seq"
    mut_col = "mut_codon_seq"
    label_col = "label"

    if wt_col not in task_data.columns or mut_col not in task_data.columns:
        logger.error(f"Task {task_name} missing required columns: {wt_col}, {mut_col}")
        CodonModelLoader.release(model, device)
        return {
            "model": model_name, "task": task_name,
            "success": False, "error": "Missing wt/mut sequence columns",
        }

    df = task_data.copy()
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    wt_seqs = df[wt_col].astype(str).tolist()
    mut_seqs = df[mut_col].astype(str).tolist()
    labels = df[label_col].values.astype(float)

    is_decoder = config.architecture in ("decoder",)
    compute_llr = compute_llr_decoder if is_decoder else compute_llr_encoder

    llrs = []
    t0 = time.time()
    for i, (wt, mut) in enumerate(zip(wt_seqs, mut_seqs)):
        try:
            llr = compute_llr(model, tokenizer, wt, mut, device=device)
            llrs.append(llr)
        except Exception as e:
            logger.warning(f"LLR computation failed for sample {i}: {e}")
            llrs.append(0.0)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(wt_seqs) - i - 1)
            logger.info(f"  [{model_name}][{task_name}] {i+1}/{len(wt_seqs)} done, ~{remaining:.0f}s remaining")

    llrs = np.array(llrs)
    elapsed = time.time() - t0

    neg_llrs = -llrs

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(labels, neg_llrs)
        auc_pr = average_precision_score(labels, neg_llrs)
    except ValueError as e:
        logger.warning(f"AUC computation failed: {e}")
        auc_roc = float("nan")
        auc_pr = float("nan")

    try:
        patho_llrs = neg_llrs[labels == 1]
        benign_llrs = neg_llrs[labels == 0]
        if len(patho_llrs) > 0 and len(benign_llrs) > 0:
            u_stat, p_val = stats.mannwhitneyu(patho_llrs, benign_llrs, alternative="greater")
            neg_log10_p = -np.log10(p_val) if p_val > 0 else float("inf")
        else:
            u_stat, p_val, neg_log10_p = float("nan"), float("nan"), float("nan")
    except Exception:
        u_stat, p_val, neg_log10_p = float("nan"), float("nan"), float("nan")

    result = {
        "model": model_name,
        "task": task_name,
        "architecture": config.architecture,
        "params_M": config.params_M,
        "success": True,
        "n_samples": len(df),
        "n_pathogenic": int(labels.sum()),
        "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC": round(float(auc_roc), 4),
        "PR-AUC": round(float(auc_pr), 4),
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "mann_whitney_U": round(float(u_stat), 2) if not np.isnan(u_stat) else None,
        "eval_time_s": round(elapsed, 1),
        "load_meta": load_meta,
    }

    CodonModelLoader.release(model, device)
    return result


def run_zeroshot_eval(
    model_names: List[str],
    task_names: List[str],
    data_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    max_samples_per_task: int = None,
) -> List[Dict]:
    from src.data.download import prepare_all_tasks

    all_task_data = prepare_all_tasks(data_dir)

    task_map = {
        "task0_cancer": "task0_cancer",
        "task1_ddd_asd": "task1_ddd_asd",
        "task2_clinvar_missense": "task2_clinvar_missense",
        "task3_synonymous": "task3_synonymous",
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
            result = evaluate_zeroshot_single(
                model_name, task_name, all_task_data[data_key],
                device=device, max_samples=max_samples_per_task,
            )
            results.append(result)

            result_file = out_path / f"{model_name}_{task_name}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"  Result: ROC-AUC={result.get('ROC-AUC', 'N/A')}, saved to {result_file}")

    summary_file = out_path / "zeroshot_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"All zero-shot results saved to {summary_file}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default=None, help="Comma-separated model names (default: all priority 1-2)")
    parser.add_argument("--tasks", default="task0_cancer,task3_synonymous", help="Comma-separated task names")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per task")
    args = parser.parse_args()

    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        all_cfgs = CodonModelLoader.load_configs()
        model_names = [n for n, c in all_cfgs.items() if c.priority <= 2]

    task_names = [t.strip() for t in args.tasks.split(",")]
    results = run_zeroshot_eval(model_names, task_names, args.data, args.output, args.device, args.max_samples)

    print(f"\n{'='*60}")
    print(f"Zero-shot evaluation complete: {len(results)} model-task pairs")
    print(f"{'='*60}")
    for r in results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:25s} | ROC-AUC={r['ROC-AUC']:.4f} | PR-AUC={r['PR-AUC']:.4f}")
        else:
            print(f"  {r['model']:20s} | {r['task']:25s} | FAILED: {r.get('error','unknown')}")
