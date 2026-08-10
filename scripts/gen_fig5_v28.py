import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "regression_mlp_lora_summary.json", "r", encoding="utf-8") as f:
    reg_data = json.load(f)

TASKS = [
    ("task4_mrfp_expression", "mRFPExpr"),
    ("task7_fungal_expression", "FungalExpr"),
    ("task5_ecoli_proteins", "EcoliExpr"),
    ("task6_mrna_stability", "mRNAStab"),
]
MODELS = ["codonbert", "codonbert_hf", "encodon-80m"]
MODEL_LABELS = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M"]

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 5, wspace=0.35,
                       left=0.05, right=0.97, top=0.88, bottom=0.15)

# --- Panel a: Randomization ablation ---
ax_a = fig.add_subplot(gs[0])
rand_models = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M", "ESM-2"]
orig_syn = [0.849, 0.816, 0.812, 0.618]
rand_syn = [0.797, 0.828, 0.829, 0.618]
orig_mis = [0.677, 0.650, 0.674, 0.692]
rand_mis = [0.673, 0.636, 0.630, 0.692]

x = np.arange(len(rand_models))
w = 0.18
ax_a.bar(x - 1.5*w, orig_syn, w, label="Original SynPath", color="#d6604d", alpha=0.9)
ax_a.bar(x - 0.5*w, rand_syn, w, label="Randomized SynPath", color="#d6604d", alpha=0.5, hatch="//")
ax_a.bar(x + 0.5*w, orig_mis, w, label="Original MisPath", color="#4393c3", alpha=0.9)
ax_a.bar(x + 1.5*w, rand_mis, w, label="Randomized MisPath", color="#4393c3", alpha=0.5, hatch="//")
ax_a.set_xticks(x)
ax_a.set_xticklabels(rand_models, fontsize=8, rotation=25, ha="right")
ax_a.set_ylabel("MLP AUC", fontsize=10)
ax_a.set_title("a  Codon randomization", fontsize=10, fontweight="bold", loc="left")
ax_a.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.25))
ax_a.grid(True, alpha=0.3, axis="y")
ax_a.set_ylim(0, 0.9)

# --- Panels b-e: Regression tasks ---
for idx, (task_key, task_name) in enumerate(TASKS):
    ax = fig.add_subplot(gs[idx + 1])

    ridge_vals, mlp_vals = [], []
    for model in MODELS:
        entry = next((e for e in reg_data if e["model"] == model and e["task"] == task_key), None)
        if entry:
            ridge_vals.append(entry["ridge_r2"])
            mlp_vals.append(entry["mlp_r2"])
        else:
            ridge_vals.append(0)
            mlp_vals.append(0)

    x = np.arange(len(MODELS))
    w = 0.3
    bars1 = ax.bar(x - w/2, ridge_vals, w, label="Ridge", color="#d6604d", alpha=0.8)
    bars2 = ax.bar(x + w/2, mlp_vals, w, label="MLP", color="#2166ac", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("R²", fontsize=10)
    panel = chr(ord('b') + idx)
    ax.set_title(f"{panel}  {task_name}", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3, axis="y")
    if idx == 0:
        ax.legend(fontsize=8)
    ax.set_ylim(-0.55, 0.75)

fig.suptitle("Figure 5 | Signal source and applicability boundary",
             fontsize=13, fontweight="bold", y=0.98)

plt.savefig(OUT_DIR / "fig5_signal_regression.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig5_signal_regression.pdf", dpi=300, bbox_inches="tight")
print("Saved fig5_signal_regression.png and .pdf")
