"""Compute bootstrap 95% CI for key Δ_AUC comparisons using 5-fold CV results."""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

with open(RESULTS_DIR / "comprehensive_comparison.json") as f:
    data = json.load(f)

task3 = {d["model"]: d for d in data if d["task"] == "task3_synonymous"}
task2 = {d["model"]: d for d in data if d["task"] == "task2_missense"}

def bootstrap_ci_delta(folds1, folds2, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(folds1)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = np.mean(np.array(folds1)[idx]) - np.mean(np.array(folds2)[idx])
        diffs.append(d)
    diffs = np.sort(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

results = []

print("=" * 80)
print("Task 3 (Synonymous): Codon-level cLMs vs ESM-2")
print("=" * 80)

esm2_t3 = task3.get("esm2")
if esm2_t3:
    codon_clms_t3 = ["encodon-620m", "codonbert", "codontransformer", "codonbert_hf", "encodon-80m"]
    for m in codon_clms_t3:
        if m in task3:
            delta = task3[m]["auc_mean"] - esm2_t3["auc_mean"]
            ci_lo, ci_hi = bootstrap_ci_delta(task3[m]["auc_folds"], esm2_t3["auc_folds"])
            print(f"  {m} vs ESM-2: Δ_AUC = {delta:+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
            results.append({
                "comparison": f"{m} vs ESM-2",
                "task": "task3_synonymous",
                "delta_auc": round(delta, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
            })

print()
print("=" * 80)
print("Task 2 (Missense): ESM-2 vs best cLM")
print("=" * 80)

esm2_t2 = task2.get("esm2")
if esm2_t2:
    best_clm_t2 = max(
        [(m, task2[m]) for m in task2 if task2[m].get("type") == "cLM"],
        key=lambda x: x[1]["auc_mean"]
    )
    delta = esm2_t2["auc_mean"] - best_clm_t2[1]["auc_mean"]
    ci_lo, ci_hi = bootstrap_ci_delta(esm2_t2["auc_folds"], best_clm_t2[1]["auc_folds"])
    print(f"  ESM-2 vs {best_clm_t2[0]}: Δ_AUC = {delta:+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    results.append({
        "comparison": f"ESM-2 vs {best_clm_t2[0]}",
        "task": "task2_missense",
        "delta_auc": round(delta, 4),
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
    })

print()
print("=" * 80)
print("Task 3: Codon-level cLMs vs ESM-1b")
print("=" * 80)

esm1b_t3 = task3.get("esm1b")
if esm1b_t3:
    for m in ["codonbert", "codonbert_hf", "encodon-80m"]:
        if m in task3:
            delta = task3[m]["auc_mean"] - esm1b_t3["auc_mean"]
            ci_lo, ci_hi = bootstrap_ci_delta(task3[m]["auc_folds"], esm1b_t3["auc_folds"])
            print(f"  {m} vs ESM-1b: Δ_AUC = {delta:+.3f}, 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
            results.append({
                "comparison": f"{m} vs ESM-1b",
                "task": "task3_synonymous",
                "delta_auc": round(delta, 4),
                "ci_95_lo": round(ci_lo, 4),
                "ci_95_hi": round(ci_hi, 4),
            })

print()
print("=" * 80)
print("Task 3: BERT family tokenization ablation (codon vs char)")
print("=" * 80)

codon_bert_mean_t3 = np.mean([task3[m]["auc_mean"] for m in ["codonbert", "codonbert_hf"] if m in task3])
char_bert_mean_t3 = np.mean([task3[m]["auc_mean"] for m in ["cdsbert", "cdsbert_plus"] if m in task3])
print(f"  Codon-BERT mean: {codon_bert_mean_t3:.3f}")
print(f"  Char-BERT mean:  {char_bert_mean_t3:.3f}")
print(f"  Δ = {codon_bert_mean_t3 - char_bert_mean_t3:+.3f}")

with open(RESULTS_DIR / "bootstrap_ci_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'bootstrap_ci_results.json'}")