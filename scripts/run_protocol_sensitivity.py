"""P0-2: Protocol sensitivity analysis.

Compute Spearman rank correlations across probing protocols (LR, MLP, LoRA)
for both tasks. Key question: does the probing protocol change model rankings?

If rankings are unstable (low Spearman), single-protocol evaluation is misleading.
If rankings are stable (high Spearman), protocol choice doesn't matter much.
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("./results")

def load_all_results():
    """Load results from all JSON files into unified format."""
    data = {}

    # 3 core cLMs (LR CV + MLP test + LoRA test)
    with open(RESULTS_DIR / "cLM_mlp_independent_results.json") as f:
        for r in json.load(f):
            key = r["model"]
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {
                "lr": r["cv_lr_mean"],
                "mlp": r["test_mlp"],
            }

    with open(RESULTS_DIR / "lora_finetune_results.json") as f:
        for r in json.load(f):
            key = r["model"]
            task = r["task"]
            if key in data and task in data[key]:
                data[key][task]["lora"] = r["test_lora_auc"]

    # EnCodon-620M
    with open(RESULTS_DIR / "encodon-620m_task2_missense.json") as f:
        r = json.load(f)
        if "encodon-620m" not in data:
            data["encodon-620m"] = {}
        data["encodon-620m"]["task2_missense"] = {
            "lr": r["lr_auc_mean"],
            "mlp": r["mlp_auc_test"],
        }
    with open(RESULTS_DIR / "encodon-620m_task3_synonymous.json") as f:
        r = json.load(f)
        data["encodon-620m"]["task3_synonymous"] = {
            "lr": r["lr_auc_mean"],
            "mlp": r["mlp_auc_test"],
        }
    with open(RESULTS_DIR / "lora_extended_results.json") as f:
        for r in json.load(f):
            if r["model"] == "encodon-620m" and r.get("success"):
                data["encodon-620m"][r["task"]]["lora"] = r["test_auc"]

    # CodonTransformer
    with open(RESULTS_DIR / "codontransformer_summary.json") as f:
        for r in json.load(f):
            key = "codontransformer"
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {
                "lr": r["lr_auc_mean"],
                "mlp": r["mlp_auc_test"],
            }
    with open(RESULTS_DIR / "lora_extended_results.json") as f:
        for r in json.load(f):
            if r["model"] == "codontransformer" and r.get("success"):
                data["codontransformer"][r["task"]]["lora"] = r["test_auc"]

    # CaLM
    with open(RESULTS_DIR / "calm_summary.json") as f:
        for r in json.load(f):
            key = "calm"
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {
                "lr": r["lr_auc_mean"],
                "mlp": r["mlp_auc_test"],
            }

    # cdsBERT (char-level)
    with open(RESULTS_DIR / "cdsbert_char_summary.json") as f:
        for r in json.load(f):
            if r["model"] == "cdsBERT":
                key = "cdsbert"
                if key not in data:
                    data[key] = {}
                task = r["task"]
                data[key][task] = {
                    "lr": r["lr_auc_mean"],
                    "mlp": r["mlp_auc_test"],
                }

    # Mistral-Codon (117M, 16M, 1M)
    with open(RESULTS_DIR / "mistral_codon_summary.json") as f:
        for r in json.load(f):
            key = r["model"].replace("-", "_")
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {
                "lr": r["lr_auc_mean"],
                "mlp": r["mlp_auc_test"],
            }

    # ESM-2 (pLM, LR only for now)
    with open(RESULTS_DIR / "esm2_task2.json") as f:
        r = json.load(f)
        data["esm2"] = {"task2_missense": {"lr": r["ROC-AUC_mean"]}}
    with open(RESULTS_DIR / "esm2_task3.json") as f:
        r = json.load(f)
        data["esm2"]["task3_synonymous"] = {"lr": r["ROC-AUC_mean"]}

    # ESM-1b (pLM, LR only for now)
    with open(RESULTS_DIR / "esm1b_results.json", encoding="utf-16") as f:
        r = json.load(f)
        data["esm1b"] = {
            "task2_missense": {"lr": r["task2_missense"]["ROC-AUC_mean"]},
            "task3_synonymous": {"lr": r["task3_synonymous"]["ROC-AUC_mean"]},
        }

    # NT-50M, NT-500M (DNA LM, LR only)
    with open(RESULTS_DIR / "dna_lm_baselines_summary.json") as f:
        for r in json.load(f):
            if not r.get("success"):
                continue
            key = r["model"].replace("-", "_")
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {"lr": r["ROC-AUC_mean"]}

    # Traditional baselines (LR only)
    with open(RESULTS_DIR / "traditional_baselines_summary.json") as f:
        for r in json.load(f):
            key = r["model"]
            if key not in data:
                data[key] = {}
            task = r["task"]
            data[key][task] = {"lr": r["ROC-AUC_mean"]}

    return data

def compute_rank_correlations(data, task, protocols=("lr", "mlp", "lora")):
    """Compute pairwise Spearman rank correlations between protocols for a task."""
    models_with_all = []
    for model_name in sorted(data.keys()):
        if task in data[model_name]:
            available = [p for p in protocols if p in data[model_name][task]]
            if len(available) >= 2:
                models_with_all.append(model_name)

    if not models_with_all:
        return {}

    results = {}
    available_protocols = set()
    for m in models_with_all:
        available_protocols.update(data[m][task].keys())
    available_protocols = sorted(available_protocols & set(protocols))

    for i, p1 in enumerate(available_protocols):
        for p2 in available_protocols[i+1:]:
            common_models = [m for m in models_with_all
                             if p1 in data[m][task] and p2 in data[m][task]]
            if len(common_models) < 3:
                continue
            vals1 = [data[m][task][p1] for m in common_models]
            vals2 = [data[m][task][p2] for m in common_models]
            rho, pval = stats.spearmanr(vals1, vals2)
            results[f"{p1}_vs_{p2}"] = {
                "spearman_rho": round(rho, 4),
                "p_value": round(pval, 4),
                "n_models": len(common_models),
                "models": common_models,
                "vals_1": [round(v, 4) for v in vals1],
                "vals_2": [round(v, 4) for v in vals2],
            }

    return results

def print_rankings(data, task, protocol):
    """Print model rankings for a given task and protocol."""
    models = []
    for m in sorted(data.keys()):
        if task in data[m] and protocol in data[m][task]:
            models.append((m, data[m][task][protocol]))
    models.sort(key=lambda x: x[1], reverse=True)
    print(f"  {protocol.upper()} rankings ({task}):")
    for rank, (m, v) in enumerate(models, 1):
        print(f"    {rank:2d}. {m:25s} {v:.4f}")
    return [m for m, _ in models]

def main():
    data = load_all_results()

    print("=" * 70)
    print("PROTOCOL SENSITIVITY ANALYSIS")
    print("=" * 70)

    for task in ["task2_missense", "task3_synonymous"]:
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print(f"{'='*70}")

        for protocol in ["lr", "mlp", "lora"]:
            print_rankings(data, task, protocol)

        correlations = compute_rank_correlations(data, task)
        print(f"\n  Spearman rank correlations:")
        for pair, info in correlations.items():
            sig = "***" if info["p_value"] < 0.001 else "**" if info["p_value"] < 0.01 else "*" if info["p_value"] < 0.05 else "n.s."
            print(f"    {pair}: rho={info['spearman_rho']:.4f}, p={info['p_value']:.4f} {sig} (n={info['n_models']} models)")
            print(f"      Models: {info['models']}")
            print(f"      {pair.split('_vs_')[0].upper()}: {info['vals_1']}")
            print(f"      {pair.split('_vs_')[1].upper()}: {info['vals_2']}")

    print(f"\n{'='*70}")
    print("KEY FINDING: Channel-specificity of protocol sensitivity")
    print("=" * 70)
    print("""
    If LR↔MLP Spearman is LOW on SynPath but HIGH on MisPath:
      → Protocol choice specifically affects the synonymous channel evaluation
      → Channel-specificity of probing-depth dependency is CONFIRMED

    If LR↔MLP Spearman is HIGH on both tasks:
      → Protocol choice doesn't change rankings → claim weakened
    """)

    # Save results
    output = {}
    for task in ["task2_missense", "task3_synonymous"]:
        output[task] = compute_rank_correlations(data, task)
    with open(RESULTS_DIR / "protocol_sensitivity.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_DIR / 'protocol_sensitivity.json'}")

if __name__ == "__main__":
    main()