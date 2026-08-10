import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
CKA_FILE = BASE / "results" / "cka_extended_results.json"
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CKA_FILE, "r", encoding="utf-8") as f:
    cka_data = json.load(f)

MODELS = ["encodon-620m", "encodon-80m", "codonbert", "codonbert_hf", "codontransformer", "calm"]
LABELS = ["EnCodon\n620M", "EnCodon\n80M", "CodonBERT", "CodonBERT\n-HF", "CodonTrans\nformer", "CaLM"]
N = len(MODELS)

def build_matrix(metric):
    mat = np.eye(N)
    for entry in cka_data:
        m1, m2 = entry["model1"], entry["model2"]
        if m1 in MODELS and m2 in MODELS:
            i, j = MODELS.index(m1), MODELS.index(m2)
            val = entry[metric]
            mat[i, j] = val
            mat[j, i] = val
    return mat

lin_mat = build_matrix("linear_cka")
rbf_mat = build_matrix("rbf_cka")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, mat, title in [(ax1, lin_mat, "Linear CKA"), (ax2, rbf_mat, "RBF CKA")]:
    vmin = 0.3 if "Linear" in title else 0.8
    im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=1.0, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(LABELS, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("Supplementary Figure S6: CKA similarity hierarchy", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "figS6_cka_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS6_cka_heatmap.pdf", bbox_inches="tight")
print("Saved figS6_cka_heatmap.png and .pdf")