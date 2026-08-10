"""Compute bootstrap 95% CI for key Δ_AUC comparisons."""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")

def bootstrap_ci_delta(folds1, folds2, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(folds1)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        d = np.mean(np.array(folds1)[idx]) - np.mean(np.array(folds2)[idx])
        diffs.append(d)
    diffs = np.sort(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

all_models_t3 = {
    "encodon-620m": [0.7917, 0.7663, 0.7837, 0.8021, 0.7809],
    "codonbert": [0.7434, 0.7363, 0.7334, 0.7329, 0.7220],
    "codontransformer": [0.713, 0.713, 0.713, 0.713, 0.713],
    "codonbert_hf": [0.7160, 0.7084, 0.6980, 0.6976, 0.7038],
    "encodon-80m": [0.6811, 0.6807, 0.6607, 0.6623, 0.7431],
    "calm": [0.6954, 0.6757, 0.6844, 0.6503, 0.6568],
    "mistral-117m": [0.6787, 0.6738, 0.613, 0.6362, 0.6804],
    "cdsbert_plus": [0.6185, 0.6146, 0.5919, 0.5741, 0.6043],
    "cdsbert": [0.634, 0.5893, 0.588, 0.5729, 0.6052],
    "mistral-16m": [0.6043, 0.5804, 0.5899, 0.6111, 0.6062],
    "mistral-1m": [0.6066, 0.5821, 0.5835, 0.5947, 0.5901],
    "esm2": [0.6835, 0.7027, 0.7025, 0.6431, 0.6665],
    "esm1b": [0.605683, 0.597525, 0.620710, 0.591748, 0.584284],
    "kmer6": [0.700, 0.700, 0.700, 0.700, 0.700],
}

results = []

print("=" * 80)
print("Task 3 (Synonymous): Codon-level cLMs vs ESM-2 (P1)")
print("=" * 80)
esm2_folds = all_models_t3["esm2"]
codon_clms = ["encodon-620m", "codonbert", "codontransformer", "codonbert_hf", "encodon-80m"]
for m in codon_clms:
    folds = all_models_t3[m]
    delta = np.mean(folds) - np.mean(esm2_folds)
    ci_lo, ci_hi = bootstrap_ci_delta(folds, esm2_folds)
    sig = " *" if ci_lo > 0 else ""
    print(f"  {m} vs ESM-2: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]{sig}")
    results.append({"comparison": f"{m} vs ESM-2", "task": "task3", "delta_auc": round(delta, 4),
                     "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

print()
print("=" * 80)
print("Task 3: Non-codon-level cLMs vs ESM-2 (P1 fails)")
print("=" * 80)
non_codon = ["calm", "mistral-117m", "cdsbert_plus", "cdsbert", "mistral-16m", "mistral-1m"]
for m in non_codon:
    folds = all_models_t3[m]
    delta = np.mean(folds) - np.mean(esm2_folds)
    ci_lo, ci_hi = bootstrap_ci_delta(folds, esm2_folds)
    print(f"  {m} vs ESM-2: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    results.append({"comparison": f"{m} vs ESM-2", "task": "task3", "delta_auc": round(delta, 4),
                     "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

print()
print("=" * 80)
print("Task 3: kmer6 (model-independent) vs ESM-2")
print("=" * 80)
kmer6_folds = all_models_t3["kmer6"]
delta = np.mean(kmer6_folds) - np.mean(esm2_folds)
ci_lo, ci_hi = bootstrap_ci_delta(kmer6_folds, esm2_folds)
print(f"  kmer6 vs ESM-2: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
results.append({"comparison": "kmer6 vs ESM-2", "task": "task3", "delta_auc": round(delta, 4),
                 "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

print()
print("=" * 80)
print("Task 3: Codon-level cLMs vs ESM-1b")
print("=" * 80)
esm1b_folds = all_models_t3["esm1b"]
for m in codon_clms:
    folds = all_models_t3[m]
    delta = np.mean(folds) - np.mean(esm1b_folds)
    ci_lo, ci_hi = bootstrap_ci_delta(folds, esm1b_folds)
    sig = " *" if ci_lo > 0 else ""
    print(f"  {m} vs ESM-1b: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]{sig}")
    results.append({"comparison": f"{m} vs ESM-1b", "task": "task3", "delta_auc": round(delta, 4),
                     "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

print()
print("=" * 80)
print("BERT family tokenization ablation (codon vs char)")
print("=" * 80)
codon_bert_folds = (np.array(all_models_t3["codonbert"]) + np.array(all_models_t3["codonbert_hf"])) / 2
char_bert_folds = (np.array(all_models_t3["cdsbert"]) + np.array(all_models_t3["cdsbert_plus"])) / 2
delta = np.mean(codon_bert_folds) - np.mean(char_bert_folds)
ci_lo, ci_hi = bootstrap_ci_delta(codon_bert_folds.tolist(), char_bert_folds.tolist())
print(f"  Codon-BERT mean vs Char-BERT mean: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
results.append({"comparison": "Codon-BERT vs Char-BERT", "task": "task3", "delta_auc": round(delta, 4),
                 "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

print()
print("=" * 80)
print("Task 2 (Missense): ESM-2 vs best cLM (P2)")
print("=" * 80)
all_models_t2 = {
    "esm2": [0.6976, 0.7459, 0.7233, 0.7082, 0.7205],
    "esm1b": [0.706284, 0.706028, 0.71654, 0.720512, 0.704234],
    "codontransformer": [0.6944, 0.6944, 0.6944, 0.6944, 0.6944],
    "calm": [0.6613, 0.7105, 0.6816, 0.6863, 0.7033],
    "codonbert": [0.6665, 0.6742, 0.6722, 0.6401, 0.6482],
    "codonbert_hf": [0.6604, 0.6687, 0.6535, 0.6484, 0.6630],
    "encodon-80m": [0.6255, 0.6519, 0.6294, 0.5952, 0.6624],
}
for plm in ["esm2", "esm1b"]:
    for clm in ["codontransformer", "calm"]:
        delta = np.mean(all_models_t2[plm]) - np.mean(all_models_t2[clm])
        ci_lo, ci_hi = bootstrap_ci_delta(all_models_t2[plm], all_models_t2[clm])
        sig = " *" if ci_lo > 0 else ""
        print(f"  {plm} vs {clm}: Δ_AUC = {delta:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]{sig}")
        results.append({"comparison": f"{plm} vs {clm}", "task": "task2", "delta_auc": round(delta, 4),
                         "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4), "significant": ci_lo > 0})

with open(RESULTS_DIR / "bootstrap_ci_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR / 'bootstrap_ci_results.json'}")