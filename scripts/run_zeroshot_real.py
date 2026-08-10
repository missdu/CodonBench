import os
import sys
import json
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder, compute_llr_decoder
from src.data.download import prepare_task2, prepare_task3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/zeroshot_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_EVAL = ["encodon-80m", "codonbert"]
TASKS_TO_EVAL = ["task2_clinvar_missense", "task3_synonymous"]
MAX_SAMPLES = 500


def load_task_data(task_name):
    if task_name == "task2_clinvar_missense":
        return prepare_task2(DATA_DIR)
    elif task_name == "task3_synonymous":
        return prepare_task3(DATA_DIR)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def evaluate_model_task(model_name, task_name, task_data):
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, load_meta = CodonModelLoader.load(model_name, device=DEVICE)

    if model is None:
        logger.error(f"Cannot load {model_name}: {load_meta.get('error')}")
        return {"model": model_name, "task": task_name, "success": False, "error": load_meta.get("error")}

    df = task_data.copy()
    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)

    wt_seqs = df["wt_codon_seq"].astype(str).tolist()
    mut_seqs = df["mut_codon_seq"].astype(str).tolist()
    labels = df["label"].values.astype(float)

    is_decoder = config.architecture in ("decoder",)
    compute_llr = compute_llr_decoder if is_decoder else compute_llr_encoder

    llrs = []
    t0 = time.time()
    for i, (wt, mut) in enumerate(zip(wt_seqs, mut_seqs)):
        try:
            llr = compute_llr(model, tokenizer, wt, mut, device=DEVICE)
            llrs.append(llr)
        except Exception as e:
            logger.warning(f"LLR failed sample {i}: {e}")
            llrs.append(0.0)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(wt_seqs) - i - 1)
            logger.info(f"  [{model_name}][{task_name}] {i+1}/{len(wt_seqs)}, ~{remaining:.0f}s left")

    llrs = np.array(llrs)
    elapsed = time.time() - t0
    neg_llrs = -llrs

    try:
        auc_roc = roc_auc_score(labels, neg_llrs)
        auc_pr = average_precision_score(labels, neg_llrs)
    except ValueError as e:
        logger.warning(f"AUC failed: {e}")
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

    CodonModelLoader.release(model, DEVICE)

    result_file = OUT_DIR / f"{model_name}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved: {result_file}")
    return result


def main():
    task_data_cache = {}
    all_results = []

    for model_name in MODELS_TO_EVAL:
        for task_name in TASKS_TO_EVAL:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_name} on {task_name}")
            logger.info(f"{'='*60}")

            if task_name not in task_data_cache:
                task_data_cache[task_name] = load_task_data(task_name)

            try:
                result = evaluate_model_task(model_name, task_name, task_data_cache[task_name])
                all_results.append(result)
                if result.get("success"):
                    logger.info(f"  ROC-AUC={result['ROC-AUC']:.4f}, PR-AUC={result['PR-AUC']:.4f}")
                else:
                    logger.error(f"  FAILED: {result.get('error')}")
            except Exception as e:
                logger.error(f"  Exception: {e}")
                all_results.append({"model": model_name, "task": task_name, "success": False, "error": str(e)})

    summary_file = OUT_DIR / "zeroshot_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Zero-shot evaluation complete: {len(all_results)} model-task pairs")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:30s} | ROC-AUC={r['ROC-AUC']:.4f} | PR-AUC={r['PR-AUC']:.4f} | time={r['eval_time_s']:.0f}s")
        else:
            print(f"  {r['model']:20s} | {r['task']:30s} | FAILED: {r.get('error','unknown')}")


if __name__ == "__main__":
    main()