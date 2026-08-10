import json
import numpy as np
from pathlib import Path
from scipy import stats

def delong_roc_test(y_true, y_pred1, y_pred2):
    n1 = np.sum(y_true == 1)
    n2 = np.sum(y_true == 0)
    
    sorted_idx = np.argsort(-y_pred1)
    y_true_sorted = y_true[sorted_idx]
    y_pred1_sorted = y_pred1[sorted_idx]
    y_pred2_sorted = y_pred2[sorted_idx]
    
    tpr1 = np.cumsum(y_true_sorted) / n1
    fpr1 = np.cumsum(1 - y_true_sorted) / n2
    
    auc1 = np.trapz(tpr1, fpr1)
    
    sorted_idx2 = np.argsort(-y_pred2)
    y_true_sorted2 = y_true[sorted_idx2]
    y_pred2_sorted2 = y_pred2[sorted_idx2]
    
    tpr2 = np.cumsum(y_true_sorted2) / n1
    fpr2 = np.cumsum(1 - y_true_sorted2) / n2
    
    auc2 = np.trapz(tpr2, fpr2)
    
    from sklearn.metrics import roc_auc_score
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    
    return auc1, auc2

def paired_ttest_on_folds(auc_folds_a, auc_folds_b):
    if len(auc_folds_a) != len(auc_folds_b):
        return None, None
    diff = np.array(auc_folds_a) - np.array(auc_folds_b)
    if np.std(diff) == 0:
        return 0.0, 1.0
    t_stat, p_val = stats.ttest_rel(auc_folds_a, auc_folds_b)
    return t_stat, p_val

def main():
    results_dir = Path("./results")
    
    all_results = []
    
    cLM_file = results_dir / "downstream_cds_full_v2_summary.json"
    if cLM_file.exists():
        cLM_results = json.load(open(cLM_file))
        for r in cLM_results:
            if r.get("success"):
                all_results.append({
                    "model": r["model"],
                    "task": r["task"].replace("_cds_full", "").replace("_full", ""),
                    "type": "cLM",
                    "auc_mean": r["ROC-AUC_mean"],
                    "auc_std": r["ROC-AUC_std"],
                    "auc_folds": r.get("ROC-AUC_folds", []),
                    "n_samples": r["n_samples"],
                    "params_M": r.get("params_M", 0),
                })
    
    dna_file = results_dir / "dna_lm_baselines_summary.json"
    if dna_file.exists():
        dna_results = json.load(open(dna_file))
        for r in dna_results:
            if r.get("success"):
                all_results.append({
                    "model": r["model"],
                    "task": r["task"],
                    "type": "DNA LM",
                    "auc_mean": r["ROC-AUC_mean"],
                    "auc_std": r["ROC-AUC_std"],
                    "auc_folds": r.get("ROC-AUC_folds", []),
                    "n_samples": r["n_samples"],
                    "params_M": r.get("params_M", 0),
                })
    
    trad_file = results_dir / "traditional_baselines_summary.json"
    if trad_file.exists():
        trad_results = json.load(open(trad_file))
        for r in trad_results:
            if r.get("success"):
                all_results.append({
                    "model": r["model"],
                    "task": r["task"],
                    "type": "Traditional",
                    "auc_mean": r["ROC-AUC_mean"],
                    "auc_std": r["ROC-AUC_std"],
                    "auc_folds": r.get("ROC-AUC_folds", []),
                    "n_samples": r["n_samples"],
                    "params_M": 0,
                })
    
    print("=" * 80)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("=" * 80)
    
    for task in ["task2_missense", "task3_synonymous"]:
        task_results = [r for r in all_results if r["task"] == task or task in r["task"]]
        task_results.sort(key=lambda x: x["auc_mean"], reverse=True)
        
        print(f"\n--- {task} ---")
        print(f"{'Model':20s} {'Type':12s} {'AUC':>10s} {'Std':>8s} {'n':>6s} {'Params':>8s}")
        print("-" * 70)
        for r in task_results:
            print(f"{r['model']:20s} {r['type']:12s} {r['auc_mean']:10.4f} {r['auc_std']:8.4f} {r['n_samples']:6d} {r['params_M']:7.0f}M")
    
    print(f"\n{'=' * 80}")
    print("PAIRED t-TESTS (within same task, same folds)")
    print("=" * 80)
    
    for task in ["task2_missense", "task3_synonymous"]:
        task_results = [r for r in all_results if r["task"] == task or task in r["task"]]
        task_results.sort(key=lambda x: x["auc_mean"], reverse=True)
        
        print(f"\n--- {task} ---")
        for i in range(len(task_results)):
            for j in range(i+1, len(task_results)):
                a = task_results[i]
                b = task_results[j]
                if a["auc_folds"] and b["auc_folds"] and len(a["auc_folds"]) == len(b["auc_folds"]):
                    t_stat, p_val = paired_ttest_on_folds(a["auc_folds"], b["auc_folds"])
                    sig = ""
                    if p_val is not None:
                        if p_val < 0.001: sig = "***"
                        elif p_val < 0.01: sig = "**"
                        elif p_val < 0.05: sig = "*"
                    p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
                    print(f"  {a['model']:20s} vs {b['model']:20s}: diff={a['auc_mean']-b['auc_mean']:+.4f}, p={p_str} {sig}")
    
    out_file = results_dir / "comprehensive_comparison.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()