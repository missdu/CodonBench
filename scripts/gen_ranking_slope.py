import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "lr_mlp_full_comparison.json", "r", encoding="utf-8") as f:
    data = json.load(f)

TOKEN_COLORS = {"codon": "#2166ac", "char": "#e08214", "protein": "#1a9850"}
TOKEN_TYPES = {
    "encodon-620m": "codon", "codonbert": "codon", "codontransformer": "codon",
    "codonbert_hf": "codon", "encodon-80m": "codon", "mistralcodon117m": "codon",
    "mistralcodon16m": "codon", "mistralcodon1m": "codon", "calm": "codon",
    "cdsbert": "char", "esm2_650m": "protein", "esm1b": "protein"
}
LABELS = {
    "encodon-620m": "EnCodon-620M", "codonbert": "CodonBERT", "codontransformer": "CodonTrans.",
    "codonbert_hf": "CodonBERT-HF", "encodon-80m": "EnCodon-80M", "mistralcodon117m": "Mistral-117M",
    "mistralcodon16m": "Mistral-16M", "mistralcodon1m": "Mistral-1M", "calm": "CaLM",
    "cdsbert": "cdsBERT", "esm2_650m": "ESM-2", "esm1b": "ESM-1b"
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax_idx, task in enumerate(["task3_synonymous", "task2_missense"]):
    ax = axes[ax_idx]
    task_data = sorted([d for d in data if d["task"] == task], key=lambda x: -x["lr_auc"])
    n = len(task_data)

    lr_ranks = list(range(1, n + 1))
    mlp_sorted = sorted(task_data, key=lambda x: -x["mlp_auc"])
    model_to_mlp_rank = {d["model"]: r + 1 for r, d in enumerate(mlp_sorted)}

    for i, d in enumerate(task_data):
        m = d["model"]
        lr_rank = i + 1
        mlp_rank = model_to_mlp_rank[m]
        color = TOKEN_COLORS[TOKEN_TYPES[m]]
        lw = 2.5 if abs(lr_rank - mlp_rank) >= 3 else 1.8
        alpha = 0.9 if TOKEN_TYPES[m] == "codon" else 0.65
        ax.plot([0, 1], [lr_rank, mlp_rank], color=color, linewidth=lw, alpha=alpha, zorder=1)
        ax.scatter([0], [lr_rank], color=color, s=70, zorder=2, edgecolors='white', linewidths=0.5)
        ax.scatter([1], [mlp_rank], color=color, s=70, zorder=2, edgecolors='white', linewidths=0.5)

    for i, d in enumerate(task_data):
        m = d["model"]
        lr_rank = i + 1
        mlp_rank = model_to_mlp_rank[m]
        color = TOKEN_COLORS[TOKEN_TYPES[m]]
        gain = d["mlp_auc"] - d["lr_auc"]
        gain_str = f"{'+'if gain>=0 else ''}{gain*100:.1f}pp"
        label_lr = f"{LABELS[m]} ({d['lr_auc']:.3f})"
        label_mlp = f"{LABELS[m]} ({d['mlp_auc']:.3f}, {gain_str})"
        ax.text(-0.12, lr_rank, label_lr, ha="right", va="center", fontsize=7.5,
                color=color, fontweight="bold")
        ax.text(1.15, mlp_rank, label_mlp, ha="left", va="center", fontsize=7.5,
                color=color, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LR", "MLP"], fontsize=13, fontweight="bold")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(range(1, n + 1), fontsize=9)
    ax.set_ylabel("Rank (1 = best)", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 2.1)
    ax.grid(True, alpha=0.2, axis="y")

    title = "SynPath" if task == "task3_synonymous" else "MisPath"
    ax.set_title(title, fontsize=14, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=TOKEN_COLORS["codon"], label="Codon-tokenized cLM (n=9)"),
    Patch(facecolor=TOKEN_COLORS["char"], label="Character-tokenized (n=1)"),
    Patch(facecolor=TOKEN_COLORS["protein"], label="Protein-level pLM (n=2)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("LR vs MLP ranking: 12 models, two tasks", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(OUT_DIR / "fig3_lr_mlp_ranking_slope.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig3_lr_mlp_ranking_slope.pdf", bbox_inches="tight")
print("Saved fig3_lr_mlp_ranking_slope.png and .pdf")

# Print clean tables
for task, task_name in [("task3_synonymous", "SynPath"), ("task2_missense", "MisPath")]:
    print(f"\n{'='*80}")
    print(f"  {task_name}: LR vs MLP Rankings (rank 1 = best)")
    print(f"{'='*80}")
    task_data = sorted([d for d in data if d["task"] == task], key=lambda x: -x["lr_auc"])
    mlp_sorted = sorted([d for d in data if d["task"] == task], key=lambda x: -x["mlp_auc"])
    model_to_lr_rank = {d["model"]: r + 1 for r, d in enumerate(task_data)}
    model_to_mlp_rank = {d["model"]: r + 1 for r, d in enumerate(mlp_sorted)}
    print(f"  {'Model':<20s} {'Token':<8s} {'LR AUC':>8s} {'MLP AUC':>8s} {'Gain':>8s} {'LR Rk':>6s} {'MLP Rk':>6s} {'Rk Δ':>6s}")
    for d in task_data:
        m = d["model"]
        lr_r = model_to_lr_rank[m]
        mlp_r = model_to_mlp_rank[m]
        gain = d["mlp_auc"] - d["lr_auc"]
        print(f"  {LABELS[m]:<20s} {TOKEN_TYPES[m]:<8s} {d['lr_auc']:>8.4f} {d['mlp_auc']:>8.4f} {gain:>+8.4f} {lr_r:>6d} {mlp_r:>6d} {mlp_r - lr_r:>+6d}")
