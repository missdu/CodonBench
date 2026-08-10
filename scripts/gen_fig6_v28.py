import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "cka_extended_results.json", "r", encoding="utf-8") as f:
    cka_data = json.load(f)
with open(BASE / "results" / "biological_findings_results.json", "r", encoding="utf-8") as f:
    bio_data = json.load(f)

MODELS_CKA = ["encodon-620m", "encodon-80m", "codonbert", "codonbert_hf", "codontransformer", "calm"]
LABELS_CKA = ["EnCodon\n620M", "EnCodon\n80M", "CodonBERT", "CodonBERT\n-HF", "CodonTrans.", "CaLM"]
N = len(MODELS_CKA)

def build_matrix(metric):
    mat = np.full((N, N), np.nan)
    np.fill_diagonal(mat, 1.0)
    for entry in cka_data:
        m1, m2 = entry["model1"], entry["model2"]
        if m1 in MODELS_CKA and m2 in MODELS_CKA:
            i, j = MODELS_CKA.index(m1), MODELS_CKA.index(m2)
            val = entry[metric]
            mat[i, j] = val
            mat[j, i] = val
    return mat

lin_mat = build_matrix("linear_cka")
rbf_mat = build_matrix("rbf_cka")

fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(1, 3, wspace=0.35,
                       left=0.05, right=0.97, top=0.88, bottom=0.12)

# --- Panel a: Linear CKA heatmap ---
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(lin_mat, cmap="YlOrRd", vmin=0.3, vmax=1.0, aspect="equal")
nan_mask = np.isnan(lin_mat)
if nan_mask.any():
    ax1.imshow(nan_mask.astype(float), cmap="gray", vmin=0, vmax=1, aspect="equal", alpha=0.3)
ax1.set_xticks(range(N))
ax1.set_yticks(range(N))
ax1.set_xticklabels(LABELS_CKA, fontsize=7, rotation=45, ha="right")
ax1.set_yticklabels(LABELS_CKA, fontsize=7)
ax1.set_title("a  Linear CKA", fontsize=11, fontweight="bold", loc="left")
for i in range(N):
    for j in range(N):
        if np.isnan(lin_mat[i, j]):
            ax1.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
        else:
            val = lin_mat[i, j]
            color = "white" if val > 0.7 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# --- Panel b: RBF CKA heatmap ---
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(rbf_mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="equal")
nan_mask2 = np.isnan(rbf_mat)
if nan_mask2.any():
    ax2.imshow(nan_mask2.astype(float), cmap="gray", vmin=0, vmax=1, aspect="equal", alpha=0.3)
ax2.set_xticks(range(N))
ax2.set_yticks(range(N))
ax2.set_xticklabels(LABELS_CKA, fontsize=7, rotation=45, ha="right")
ax2.set_yticklabels(LABELS_CKA, fontsize=7)
ax2.set_title("b  RBF CKA", fontsize=11, fontweight="bold", loc="left")
for i in range(N):
    for j in range(N):
        if np.isnan(rbf_mat[i, j]):
            ax2.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
        else:
            val = rbf_mat[i, j]
            color = "white" if val > 0.9 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# --- Panel c: Attribute regression (GC3, CAI, position, pathogenicity) ---
ax3 = fig.add_subplot(gs[2])
ATTRS = ["GC3", "CAI", "position_norm", "label_pathogenic"]
ATTR_LABELS = ["GC3", "CAI", "Position", "Pathogenicity"]
MODELS_BIO = ["codonbert", "codonbert_hf", "encodon-80m"]
BIO_LABELS = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M"]
COLORS = ["#2166ac", "#4393c3", "#92c5de"]

x = np.arange(len(ATTRS))
w = 0.22
for i, (model, label, color) in enumerate(zip(MODELS_BIO, BIO_LABELS, COLORS)):
    r2_vals = [bio_data[model]["attribute_regression"][attr]["r2_mean"] for attr in ATTRS]
    ax3.bar(x + (i - 1) * w, r2_vals, w, label=label, color=color, alpha=0.85)

ax3.axhline(0, color="black", linewidth=0.5)
ax3.set_xticks(x)
ax3.set_xticklabels(ATTR_LABELS, fontsize=9)
ax3.set_ylabel("R² (Ridge, 5-fold CV)", fontsize=10)
ax3.set_title("c  Attribute regression", fontsize=11, fontweight="bold", loc="left")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

fig.suptitle("Figure 6 | Representational geometry mirrors the probing-depth hierarchy",
             fontsize=13, fontweight="bold", y=0.98)

plt.savefig(OUT_DIR / "fig6_cka_attributes.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig6_cka_attributes.pdf", bbox_inches="tight")
print("Saved fig6_cka_attributes.png and .pdf")