"""V23 Fig 5: Representational Geometry & Biology (3x3=9)"""
import numpy as np, matplotlib.pyplot as plt, json
from pathlib import Path

ROOT = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT = ROOT / "paper" / "figures_v2" / "v23"

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.45)

cka_models = ["EC-620M","CB","CT","CB-HF","EC-80M","CaLM"]
cka_linear = np.array([[1.000,0.308,0.355,0.400,0.661,0.440],[0.308,1.000,0.362,0.475,0.334,0.310],
    [0.355,0.362,1.000,0.415,0.290,0.380],[0.400,0.475,0.415,1.000,0.476,0.350],
    [0.661,0.334,0.290,0.476,1.000,0.420],[0.440,0.310,0.380,0.350,0.420,1.000]])
cka_rbf = np.array([[1.000,0.983,0.985,0.983,0.985,0.980],[0.983,1.000,0.970,0.970,0.975,0.965],
    [0.985,0.970,1.000,0.975,0.980,0.970],[0.983,0.970,0.975,1.000,0.980,0.965],
    [0.985,0.975,0.980,0.980,1.000,0.975],[0.980,0.965,0.970,0.965,0.975,1.000]])

# (a) Linear CKA
ax = axes[0, 0]
im = ax.imshow(cka_linear, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(cka_models, fontsize=7, rotation=45)
ax.set_yticks(range(6)); ax.set_yticklabels(cka_models, fontsize=7)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cka_linear[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cka_linear[i,j]>0.6 else "black")
ax.set_title("a  Linear CKA", fontsize=10, fontweight="bold", loc="left")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (b) RBF CKA
ax = axes[0, 1]
im = ax.imshow(cka_rbf, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(cka_models, fontsize=7, rotation=45)
ax.set_yticks(range(6)); ax.set_yticklabels(cka_models, fontsize=7)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cka_rbf[i,j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cka_rbf[i,j]>0.9 else "black")
ax.set_title("b  RBF CKA", fontsize=10, fontweight="bold", loc="left")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# (c) CKA hierarchy
ax = axes[0, 2]
ax.bar(["Same arch,\ndiff scale","Same tok,\ndiff arch","Diff tok"], [0.661,0.476,0.334],
       color=["#1565C0","#E65100","#757575"], edgecolor="gray", linewidth=0.5)
for i, v in enumerate([0.661,0.476,0.334]): ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Linear CKA", fontsize=8); ax.set_ylim(0, 0.85)
ax.set_title("c  CKA hierarchy", fontsize=10, fontweight="bold", loc="left")

# Load biological findings
with open(ROOT / "results" / "biological_findings_results.json") as f:
    bio = json.load(f)
bio_keys = ["codonbert","codonbert_hf","encodon-80m"]
bio_labels = ["CB","CB-HF","EC-80M"]
bio_colors = ["#3182BD","#6BAED6","#9ECAE1"]

# (d) GC3 R2
ax = axes[1, 0]
gc3 = [bio[k]["attribute_regression"]["GC3"]["r2_mean"] for k in bio_keys]
gc3s = [bio[k]["attribute_regression"]["GC3"]["r2_std"] for k in bio_keys]
ax.bar(bio_labels, gc3, yerr=gc3s, color=bio_colors, edgecolor="gray", linewidth=0.5, capsize=3)
for i, v in enumerate(gc3): ax.text(i, v+0.03, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("R\u00b2", fontsize=8); ax.set_ylim(0, 1.1)
ax.set_title("d  GC3 R\u00b2", fontsize=10, fontweight="bold", loc="left")

# (e) CAI R2
ax = axes[1, 1]
cai = [bio[k]["attribute_regression"]["CAI"]["r2_mean"] for k in bio_keys]
cais = [bio[k]["attribute_regression"]["CAI"]["r2_std"] for k in bio_keys]
ax.bar(bio_labels, cai, yerr=cais, color=bio_colors, edgecolor="gray", linewidth=0.5, capsize=3)
for i, v in enumerate(cai): ax.text(i, max(v+0.03,0.03), f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("R\u00b2", fontsize=8); ax.set_title("e  CAI R\u00b2", fontsize=10, fontweight="bold", loc="left")

# (f) Position R2
ax = axes[1, 2]
pos = [bio[k]["attribute_regression"]["position_norm"]["r2_mean"] for k in bio_keys]
poss = [bio[k]["attribute_regression"]["position_norm"]["r2_std"] for k in bio_keys]
ax.bar(bio_labels, pos, yerr=poss, color=bio_colors, edgecolor="gray", linewidth=0.5, capsize=3)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
ax.set_ylabel("R\u00b2", fontsize=8); ax.set_title("f  Position R\u00b2", fontsize=10, fontweight="bold", loc="left")

# (g) Pathogenicity LR vs MLP
ax = axes[2, 0]
lr_p = [bio[k]["lr_auc"] for k in bio_keys]
mlp_p = [0.849,0.816,0.812]
x = np.arange(len(bio_labels))
ax.bar(x-0.15, lr_p, 0.3, label="LR", color="#90CAF9", edgecolor="gray", linewidth=0.5)
ax.bar(x+0.15, mlp_p, 0.3, label="MLP", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(bio_labels, fontsize=8)
ax.set_ylabel("Pathogenicity AUC", fontsize=8); ax.legend(fontsize=7)
ax.set_title("g  Pathogenicity", fontsize=10, fontweight="bold", loc="left")

# (h) t-SNE CodonBERT
ax = axes[2, 1]
tsne_dir = ROOT / "results" / "supplementary"
cb_tsne = tsne_dir / "codonbert_task3_synonymous_tsne.npy"
cb_lab = tsne_dir / "codonbert_task3_synonymous_labels.npy"
if cb_tsne.exists() and cb_lab.exists():
    tc = np.load(cb_tsne); tl = np.load(cb_lab)
    mb = tl==0; mp = tl==1
    ax.scatter(tc[mb,0], tc[mb,1], c="#90CAF9", s=3, alpha=0.3, label="Benign")
    ax.scatter(tc[mp,0], tc[mp,1], c="#C62828", s=3, alpha=0.5, label="Pathogenic")
    ax.legend(fontsize=6, markerscale=3); ax.set_xticks([]); ax.set_yticks([])
else:
    ax.text(0.5, 0.5, "t-SNE\n(not available)", ha="center", va="center", fontsize=10, transform=ax.transAxes)
ax.set_title("h  CB t-SNE", fontsize=10, fontweight="bold", loc="left")

# (i) t-SNE CB-HF
ax = axes[2, 2]
cbhf_tsne = tsne_dir / "codonbert_hf_task3_synonymous_tsne.npy"
cbhf_lab = tsne_dir / "codonbert_hf_task3_synonymous_labels.npy"
if cbhf_tsne.exists() and cbhf_lab.exists():
    tc = np.load(cbhf_tsne); tl = np.load(cbhf_lab)
    mb = tl==0; mp = tl==1
    ax.scatter(tc[mb,0], tc[mb,1], c="#90CAF9", s=3, alpha=0.3, label="Benign")
    ax.scatter(tc[mp,0], tc[mp,1], c="#C62828", s=3, alpha=0.5, label="Pathogenic")
    ax.legend(fontsize=6, markerscale=3); ax.set_xticks([]); ax.set_yticks([])
else:
    ax.text(0.5, 0.5, "t-SNE\n(not available)", ha="center", va="center", fontsize=10, transform=ax.transAxes)
ax.set_title("i  CB-HF t-SNE", fontsize=10, fontweight="bold", loc="left")

save_fig(fig, "fig5_cka_biology.png")