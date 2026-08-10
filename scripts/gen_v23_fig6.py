"""V23 Fig 6: Regression & Context (2x3=6)"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT = ROOT / "paper" / "figures_v2" / "v23"

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
plt.subplots_adjust(wspace=0.35, hspace=0.4)

bio_colors = ["#3182BD","#6BAED6","#9ECAE1"]
reg_tasks = ["T3: mRFP\n(within)", "T4: E.coli\n(cross)", "T5: mRNA\nstability", "T6: Fungal\n(cross)"]

# (a) Regression R2
ax = axes[0, 0]
r2_data = {"CB":[0.321,-0.05,0.09,0.544], "CB-HF":[0.458,-0.12,0.12,0.563], "EC-80M":[0.201,-0.08,0.05,0.483]}
x = np.arange(len(reg_tasks)); w = 0.25
for i, (m, vals) in enumerate(r2_data.items()):
    ax.bar(x+(i-1)*w, vals, w, label=m, color=bio_colors[i], edgecolor="gray", linewidth=0.5)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(x); ax.set_xticklabels(reg_tasks, fontsize=7)
ax.set_ylabel("R\u00b2", fontsize=8); ax.legend(fontsize=7)
ax.set_title("a  Regression R\u00b2", fontsize=10, fontweight="bold", loc="left")

# (b) Regression Spearman
ax = axes[0, 1]
sp_data = {"CB":[0.603,0.190,0.310,0.710], "CB-HF":[0.684,0.250,0.280,0.734], "EC-80M":[0.614,0.220,0.240,0.701]}
for i, (m, vals) in enumerate(sp_data.items()):
    ax.bar(x+(i-1)*w, vals, w, label=m, color=bio_colors[i], edgecolor="gray", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(reg_tasks, fontsize=7)
ax.set_ylabel("Spearman \u03c1", fontsize=8); ax.legend(fontsize=7)
ax.set_title("b  Regression Spearman \u03c1", fontsize=10, fontweight="bold", loc="left")

# (c) Within-protein scatter placeholder
ax = axes[0, 2]
ax.text(0.5, 0.5, "Within-protein\n(Task 3: mRFP)\nBest R\u00b2 = 0.458\n(CodonBERT-HF)",
        ha="center", va="center", fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0"))
ax.set_title("c  Task 3 scatter", fontsize=10, fontweight="bold", loc="left"); ax.axis("off")

# (d) Fungal cross-protein placeholder
ax = axes[1, 0]
ax.text(0.5, 0.5, "Fungal cross-protein\n(Task 6)\nBest R\u00b2 = 0.563\n(CodonBERT-HF)\nFramework-predicted \u2713",
        ha="center", va="center", fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#E65100"))
ax.set_title("d  Task 6 scatter", fontsize=10, fontweight="bold", loc="left"); ax.axis("off")

# (e) Real vs synthetic CDS
ax = axes[1, 1]
cm = ["CB","CB-HF","EC-80M"]
real = [0.660,0.659,0.633]; synth = [0.557,0.548,0.534]
x2 = np.arange(len(cm))
ax.bar(x2-0.15, real, 0.3, label="Real CDS", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.bar(x2+0.15, synth, 0.3, label="Synthetic", color="#BDBDBD", edgecolor="gray", linewidth=0.5)
ax.set_xticks(x2); ax.set_xticklabels(cm, fontsize=8)
ax.set_ylabel("Task 1 AUC", fontsize=8); ax.legend(fontsize=7)
ax.set_title("e  Real vs synthetic CDS", fontsize=10, fontweight="bold", loc="left")

# (f) Zero-shot LLR
ax = axes[1, 2]
zt = ["Task 1\nMissense", "Task 2\nSynonymous"]
zr = [0.510,0.505]; zs = [0.495,0.498]
x3 = np.arange(len(zt))
ax.bar(x3-0.15, zr, 0.3, label="Real CDS", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.bar(x3+0.15, zs, 0.3, label="Synthetic", color="#BDBDBD", edgecolor="gray", linewidth=0.5)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(x3); ax.set_xticklabels(zt, fontsize=8)
ax.set_ylabel("Zero-shot AUC", fontsize=8); ax.legend(fontsize=7)
ax.set_title("f  Zero-shot LLR", fontsize=10, fontweight="bold", loc="left")

save_fig(fig, "fig6_regression_context.png")