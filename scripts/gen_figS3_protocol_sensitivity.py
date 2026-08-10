import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "comprehensive_comparison.json", "r", encoding="utf-8") as f:
    comp_data = json.load(f)

try:
    with open(BASE / "results" / "cLM_mlp_independent_results.json", "r", encoding="utf-8") as f:
        mlp_data = json.load(f)
except FileNotFoundError:
    mlp_data = []

lr_dict = {}
for entry in comp_data:
    model = entry["model"]
    task = entry["task"]
    key = (model, task)
    lr_dict[key] = entry["auc_mean"]

mlp_dict = {}
for entry in mlp_data:
    model = entry.get("model", entry.get("model_name", ""))
    task = entry.get("task", "")
    mlp_auc = entry.get("test_mlp", entry.get("mlp_auc", None))
    if mlp_auc is not None:
        key = (model, task)
        mlp_dict[key] = mlp_auc

CDS_MODELS = ["encodon-80m", "encodon-620m", "codonbert", "codonbert_hf", "codontransformer",
              "calm", "cdsbert", "mistralcodon117m", "mistralcodon16m", "mistralcodon1m"]
PLM_MODELS = ["esm2_650m", "esm1b"]

syn_lr, syn_mlp = [], []
mis_lr, mis_mlp = [], []
syn_labels, mis_labels = [], []

for model in CDS_MODELS + PLM_MODELS:
    for task, lr_list, mlp_list, labels in [
        ("task3_synonymous", syn_lr, syn_mlp, syn_labels),
        ("task2_missense", mis_lr, mis_mlp, mis_labels),
    ]:
        lr_val = lr_dict.get((model, task))
        mlp_val = mlp_dict.get((model, task))
        if lr_val is not None and mlp_val is not None:
            lr_list.append(lr_val)
            mlp_list.append(mlp_val)
            labels.append(model)

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(mis_lr, mis_mlp, c="#4393c3", marker="o", s=80, zorder=5, label="MisPath")
ax.scatter(syn_lr, syn_mlp, c="#d6604d", marker="D", s=80, zorder=5, label="SynPath")

lims = [0.45, 0.95]
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

for i, label in enumerate(syn_labels):
    ax.annotate(label, (syn_lr[i], syn_mlp[i]), fontsize=6,
                xytext=(5, 5), textcoords="offset points", alpha=0.7)
for i, label in enumerate(mis_labels):
    ax.annotate(label, (mis_lr[i], mis_mlp[i]), fontsize=6,
                xytext=(5, -10), textcoords="offset points", alpha=0.7)

if len(syn_lr) >= 3:
    rho_syn, p_syn = stats.spearmanr(syn_lr, syn_mlp)
    ax.text(0.05, 0.95, f"SynPath: ρ = {rho_syn:.3f}, p = {p_syn:.3f}",
            transform=ax.transAxes, fontsize=9, color="#d6604d", va="top")
if len(mis_lr) >= 3:
    rho_mis, p_mis = stats.spearmanr(mis_lr, mis_mlp)
    ax.text(0.05, 0.88, f"MisPath: ρ = {rho_mis:.3f}, p = {p_mis:.3f}",
            transform=ax.transAxes, fontsize=9, color="#4393c3", va="top")

ax.set_xlabel("LR AUC", fontsize=11)
ax.set_ylabel("MLP AUC", fontsize=11)
ax.set_title("Supplementary Figure S3: Protocol sensitivity", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figS3_protocol_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS3_protocol_sensitivity.pdf", bbox_inches="tight")
print("Saved figS3_protocol_sensitivity.png and .pdf")