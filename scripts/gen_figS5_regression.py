import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "regression_mlp_lora_summary.json", "r", encoding="utf-8") as f:
    reg_data = json.load(f)

TASKS = [
    ("task4_mrfp_expression", "mRFPExpr\n(within-protein)"),
    ("task5_ecoli_proteins", "EcoliExpr\n(cross-protein)"),
    ("task6_mrna_stability", "mRNAStab"),
    ("task7_fungal_expression", "FungalExpr\n(cross-species)"),
]
MODELS = ["codonbert", "codonbert_hf", "encodon-80m"]
MODEL_LABELS = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M"]
MODEL_COLORS = ["#2166ac", "#4393c3", "#92c5de"]

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

for ax, (task_key, task_label) in zip(axes, TASKS):
    ridge_vals, mlp_vals, x_pos = [], [], []
    for i, model in enumerate(MODELS):
        entry = next((e for e in reg_data if e["model"] == model and e["task"] == task_key), None)
        if entry:
            ridge_vals.append(max(entry["ridge_r2"], -0.05))
            mlp_vals.append(max(entry["mlp_r2"], -0.05))
            x_pos.append(i)

    width = 0.35
    bars1 = ax.bar([p - width/2 for p in x_pos], ridge_vals, width, label="Ridge", color="#d6604d", alpha=0.8)
    bars2 = ax.bar([p + width/2 for p in x_pos], mlp_vals, width, label="MLP", color="#2166ac", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=30, ha="right")
    ax.set_title(task_label, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.1, 0.75)

axes[0].set_ylabel("R²", fontsize=11)
axes[0].legend(fontsize=9)

fig.suptitle("Supplementary Figure S5: Regression — Ridge vs MLP comparison",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "figS5_regression_ridge_vs_mlp.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS5_regression_ridge_vs_mlp.pdf", bbox_inches="tight")
print("Saved figS5_regression_ridge_vs_mlp.png and .pdf")