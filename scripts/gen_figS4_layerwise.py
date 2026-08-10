import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "layerwise_analysis_results.json", "r", encoding="utf-8") as f:
    layer_data = json.load(f)

syn_data = [(x["layer"], x["lr_auc"], x["mlp_auc"])
            for x in layer_data
            if x["model"] == "CodonBERT-HF" and x["task"] == "task3_synonymous"
            and isinstance(x["layer"], int)]
mis_data = [(x["layer"], x["lr_auc"], x["mlp_auc"])
            for x in layer_data
            if x["model"] == "CodonBERT-HF" and x["task"] == "task2_missense"
            and isinstance(x["layer"], int)]

syn_data.sort(key=lambda x: x[0])
mis_data.sort(key=lambda x: x[0])

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for col, (data, task_name) in enumerate([(syn_data, "SynPath"), (mis_data, "MisPath")]):
    layers = [d[0] for d in data]
    lr_vals = [d[1] for d in data]
    mlp_vals = [d[2] for d in data]

    axes[0, col].plot(layers, lr_vals, "o-", color="#2166ac", label="LR", linewidth=1.5, markersize=5)
    axes[0, col].plot(layers, mlp_vals, "s-", color="#d6604d", label="MLP", linewidth=1.5, markersize=5)
    axes[0, col].fill_between(layers, lr_vals, mlp_vals, alpha=0.15, color="gray")
    axes[0, col].set_ylabel("AUC", fontsize=10)
    axes[0, col].set_title(f"{task_name}: LR vs MLP", fontsize=11, fontweight="bold")
    axes[0, col].legend(fontsize=9)
    axes[0, col].grid(True, alpha=0.3)
    axes[0, col].set_ylim(0.55, 0.90)

    gap = [m - l for l, m in zip(lr_vals, mlp_vals)]
    axes[1, col].bar(layers, gap, color=["#d6604d" if g > 0 else "#2166ac" for g in gap], alpha=0.7)
    axes[1, col].axhline(0, color="black", linewidth=0.5)
    axes[1, col].set_xlabel("Layer", fontsize=10)
    axes[1, col].set_ylabel("MLP − LR (pp)", fontsize=10)
    axes[1, col].set_title(f"{task_name}: LR-to-MLP gap", fontsize=11, fontweight="bold")
    axes[1, col].grid(True, alpha=0.3)

fig.suptitle("Supplementary Figure S4: Layer-wise analysis — CodonBERT-HF (12-layer)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "figS4_layerwise.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS4_layerwise.pdf", bbox_inches="tight")
print("Saved figS4_layerwise.png and .pdf")