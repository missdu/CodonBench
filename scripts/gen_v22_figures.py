import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import numpy as np

FIG_DIR = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2")
OUT_DIR = FIG_DIR / "v22"
OUT_DIR.mkdir(exist_ok=True)

MAX_WIDTH = 2400

def load(name):
    img = Image.open(FIG_DIR / name)
    w, h = img.size
    if w > MAX_WIDTH:
        ratio = MAX_WIDTH / w
        img = img.resize((MAX_WIDTH, int(h * ratio)), Image.LANCZOS)
    return np.array(img)

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

# Fig 1: Framework overview (same as V19, use as-is)
fig1 = load("fig1_overview_v2_extractable_depth.png")
fig, ax = plt.subplots(figsize=(7.2, 4.5))
ax.imshow(fig1)
ax.axis("off")
save(fig, "fig1_framework.png")

# Fig 2: cLM-pLM crossover + forest plot (NO AlphaMissense panel)
# Panels: a=clm_vs_plm, b=forest_plot
img_a = load("fig2_clm_vs_plm.png")
img_b = load("fig2_forest_plot.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
axes[0].imshow(img_a); axes[0].axis("off"); axes[0].set_title("a", fontsize=14, fontweight="bold", loc="left")
axes[1].imshow(img_b); axes[1].axis("off"); axes[1].set_title("b", fontsize=14, fontweight="bold", loc="left")
plt.tight_layout(w_pad=1)
save(fig, "fig2_crossover_forest.png")

# Fig 3: From-scratch ablation + CKA
# Panels: a=ablation_depth, b=cka_similarity
img_a = load("fig_ablation_depth.png")
img_b = load("fig5_cka_similarity.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1]})
axes[0].imshow(img_a); axes[0].axis("off"); axes[0].set_title("a", fontsize=14, fontweight="bold", loc="left")
axes[1].imshow(img_b); axes[1].axis("off"); axes[1].set_title("b", fontsize=14, fontweight="bold", loc="left")
plt.tight_layout(w_pad=1)
save(fig, "fig3_ablation_cka.png")

# Fig 4: LoRA + oracle + scale amplification + regression
# Panels: a=lora_hierarchy, b=oracle_ceiling, c=scale_amplification, d=regression_boundary
img_a = load("fig9_lora_hierarchy.png")
img_b = load("fig6_oracle_ceiling.png")
img_c = load("fig_scale_amplification.png")
img_d = load("fig8_regression_boundary.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes[0, 0].imshow(img_a); axes[0, 0].axis("off"); axes[0, 0].set_title("a", fontsize=14, fontweight="bold", loc="left")
axes[0, 1].imshow(img_b); axes[0, 1].axis("off"); axes[0, 1].set_title("b", fontsize=14, fontweight="bold", loc="left")
axes[1, 0].imshow(img_c); axes[1, 0].axis("off"); axes[1, 0].set_title("c", fontsize=14, fontweight="bold", loc="left")
axes[1, 1].imshow(img_d); axes[1, 1].axis("off"); axes[1, 1].set_title("d", fontsize=14, fontweight="bold", loc="left")
plt.tight_layout(w_pad=1, h_pad=1.5)
save(fig, "fig4_lora_oracle_scale_regression.png")

print("\nAll 4 figures generated in", OUT_DIR)