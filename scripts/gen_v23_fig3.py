"""V23 Fig 3: From-Scratch Ablation (3x3=9)"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT = ROOT / "paper" / "figures_v2" / "v23"

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

ABLATION = {
    "v1":{"codon_lr":0.701,"codon_lr_std":0.013,"char_lr":0.711,"char_lr_std":0.023,"codon_mlp":0.719,"char_mlp":0.627,"codon_mlp_t1":0.655,"char_mlp_t1":0.648},
    "v2":{"codon_lr":0.669,"codon_lr_std":0.011,"char_lr":0.667,"char_lr_std":0.016,"codon_mlp":0.643,"char_mlp":0.662,"codon_mlp_t1":0.648,"char_mlp_t1":0.661},
    "v3a":{"codon_lr":0.670,"codon_lr_std":0.018,"char_lr":0.647,"char_lr_std":0.026,"codon_mlp":0.720,"char_mlp":0.606,"codon_mlp_t1":0.670,"char_mlp_t1":0.634},
    "v3b":{"codon_lr":0.706,"codon_lr_std":0.013,"char_lr":0.675,"char_lr_std":0.004,"codon_mlp":0.792,"char_mlp":0.647,"codon_mlp_t1":0.674,"char_mlp_t1":0.702},
}
DELONG = {
    "v1":{"lr_delta_pp":0.08,"lr_ci":[-4.56,4.72],"lr_p":0.81,"mlp_delta_pp":7.93,"mlp_ci":[2.73,13.13],"mlp_p":0.003},
    "v3a":{"lr_delta_pp":0.88,"lr_ci":[-3.97,5.73],"lr_p":0.54,"mlp_delta_pp":11.11,"mlp_ci":[5.77,16.44],"mlp_p":0.001},
    "v3b":{"lr_delta_pp":1.28,"lr_ci":[-3.63,6.19],"lr_p":0.57,"mlp_delta_pp":13.69,"mlp_ci":[9.05,18.33],"mlp_p":0.001},
}

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.45)
CC, CH = "#2166AC", "#E08214"

# (a) LR vs MLP side-by-side for all 4 versions (the key panel)
ax = axes[0, 0]
versions = ["v1", "v2", "v3a", "v3b"]
x = np.arange(len(versions))
w = 0.18
lr_codon = [ABLATION[v]["codon_lr"] for v in versions]
lr_char = [ABLATION[v]["char_lr"] for v in versions]
mlp_codon = [ABLATION[v]["codon_mlp"] for v in versions]
mlp_char = [ABLATION[v]["char_mlp"] for v in versions]
ax.bar(x - 1.5*w, lr_codon, w, color="#90CAF9", edgecolor="gray", linewidth=0.5, label="Codon LR")
ax.bar(x - 0.5*w, lr_char, w, color="#FFCC80", edgecolor="gray", linewidth=0.5, label="Char LR")
ax.bar(x + 0.5*w, mlp_codon, w, color=CC, edgecolor="gray", linewidth=0.5, label="Codon MLP")
ax.bar(x + 1.5*w, mlp_char, w, color=CH, edgecolor="gray", linewidth=0.5, label="Char MLP")
for i, v_name in enumerate(versions):
    delta = mlp_codon[i] - mlp_char[i]
    if v_name != "v2":
        ax.annotate("", xy=(x[i]+0.5*w, mlp_codon[i]-0.008), xytext=(x[i]+1.5*w, mlp_char[i]-0.008),
                    arrowprops=dict(arrowstyle="<->", color="#C62828", lw=1.2))
        ax.text(x[i]+w, min(mlp_codon[i], mlp_char[i])-0.025, f"+{delta*100:.1f}pp",
                ha="center", fontsize=6, fontweight="bold", color="#C62828")
ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=8)
ax.set_ylabel("SynPath AUC", fontsize=8)
ax.set_ylim(0.55, 0.85)
ax.legend(fontsize=6, ncol=2, loc="upper left")
ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.5)
ax.set_title("a  LR vs MLP: the depth gap", fontsize=10, fontweight="bold", loc="left")

# (b) v1 MLP detail
ax = axes[0, 1]; v = ABLATION["v1"]
ax.bar(["Codon","Char"], [v["codon_mlp"],v["char_mlp"]], color=[CC,CH], edgecolor="gray", linewidth=0.5)
ax.text(0, v["codon_mlp"]+0.01, f"{v['codon_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.text(1, v["char_mlp"]+0.01, f"{v['char_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.annotate("", xy=(0, v["codon_mlp"]-0.02), xytext=(1, v["char_mlp"]-0.02), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(0.5, (v["codon_mlp"]+v["char_mlp"])/2-0.03, "+9.2pp\np=0.003", ha="center", fontsize=7, fontweight="bold", color="#C62828")
ax.set_ylim(0.55, 0.78); ax.set_title("b  v1 MLP (**)", fontsize=10, fontweight="bold", loc="left")

# (c) v3a MLP detail
ax = axes[0, 2]; v = ABLATION["v3a"]
ax.bar(["Codon","Char"], [v["codon_mlp"],v["char_mlp"]], color=[CC,CH], edgecolor="gray", linewidth=0.5)
ax.text(0, v["codon_mlp"]+0.01, f"{v['codon_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.text(1, v["char_mlp"]+0.01, f"{v['char_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.annotate("", xy=(0, v["codon_mlp"]-0.02), xytext=(1, v["char_mlp"]-0.02), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(0.5, (v["codon_mlp"]+v["char_mlp"])/2-0.03, "+11.4pp\np<0.001", ha="center", fontsize=7, fontweight="bold", color="#C62828")
ax.set_ylim(0.5, 0.78); ax.set_title("c  v3a MLP (***)", fontsize=10, fontweight="bold", loc="left")

# (d) v3b MLP detail
ax = axes[1, 0]; v = ABLATION["v3b"]
ax.bar(["Codon","Char"], [v["codon_mlp"],v["char_mlp"]], color=[CC,CH], edgecolor="gray", linewidth=0.5)
ax.text(0, v["codon_mlp"]+0.01, f"{v['codon_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.text(1, v["char_mlp"]+0.01, f"{v['char_mlp']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.annotate("", xy=(0, v["codon_mlp"]-0.02), xytext=(1, v["char_mlp"]-0.02), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(0.5, (v["codon_mlp"]+v["char_mlp"])/2-0.03, "+14.5pp\np<0.001", ha="center", fontsize=7, fontweight="bold", color="#C62828")
ax.set_ylim(0.55, 0.85); ax.set_title("d  v3b MLP (***)", fontsize=10, fontweight="bold", loc="left")

# (e) v2 MLP negative control
ax = axes[1, 1]; v = ABLATION["v2"]
ax.bar(["Codon","Char"], [v["codon_mlp"],v["char_mlp"]], color=["#B0B0B0",CH], edgecolor="gray", linewidth=0.5)
ax.text(0, v["codon_mlp"]+0.01, f"{v['codon_mlp']:.3f}", ha="center", fontsize=8)
ax.text(1, v["char_mlp"]+0.01, f"{v['char_mlp']:.3f}", ha="center", fontsize=8)
ax.text(0.5, max(v["codon_mlp"],v["char_mlp"])+0.03, "Synthetic\n\u0394 = \u22121.9pp", ha="center", fontsize=7, color="#757575")
ax.set_ylim(0.55, 0.75); ax.set_title("e  v2 MLP (neg ctrl)", fontsize=10, fontweight="bold", loc="left")

# (f) Scale amplification
ax = axes[1, 2]
corpus_sizes = [4842, 4842, 114119]; deltas = [9.2, 11.4, 14.5]
ci_lower = [2.73, 5.77, 9.05]; ci_upper = [13.13, 16.44, 18.33]
ci_err = [[d-l for d,l in zip(deltas,ci_lower)], [u-d for d,u in zip(deltas,ci_upper)]]
ax.errorbar(corpus_sizes, deltas, yerr=ci_err, fmt="o-", color="#2166AC", markersize=8, capsize=5, linewidth=2, markerfacecolor="white", markeredgewidth=2)
ax.set_xscale("log"); ax.set_xlabel("Corpus size (CDS)", fontsize=8)
ax.set_ylabel("\u0394 MLP AUC (codon\u2212char, pp)", fontsize=8)
ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
ax.set_title("f  Scale amplification", fontsize=10, fontweight="bold", loc="left")

# (g) Missense: no codon advantage
ax = axes[2, 0]; v = ABLATION["v3b"]
ax.bar(["Codon","Char"], [v["codon_mlp_t1"],v["char_mlp_t1"]], color=[CC,CH], edgecolor="gray", linewidth=0.5)
ax.text(0, v["codon_mlp_t1"]+0.01, f"{v['codon_mlp_t1']:.3f}", ha="center", fontsize=8)
ax.text(1, v["char_mlp_t1"]+0.01, f"{v['char_mlp_t1']:.3f}", ha="center", fontsize=8)
ax.set_ylim(0.6, 0.75); ax.set_title("g  v3b Task 1 (missense)", fontsize=10, fontweight="bold", loc="left")

# (h) DeLong CI forest
ax = axes[2, 1]
ci_versions = ["v1","v3a","v3b"]
labels_ci = []; deltas_ci = []; ci_lo = []; ci_hi = []; colors_ci = []
for ver in ci_versions:
    ds = DELONG[ver]
    labels_ci.append(f"{ver} LR"); deltas_ci.append(ds["lr_delta_pp"]); ci_lo.append(ds["lr_ci"][0]); ci_hi.append(ds["lr_ci"][1]); colors_ci.append("#90CAF9")
    labels_ci.append(f"{ver} MLP"); deltas_ci.append(ds["mlp_delta_pp"]); ci_lo.append(ds["mlp_ci"][0]); ci_hi.append(ds["mlp_ci"][1]); colors_ci.append("#1565C0")
for i in range(len(labels_ci)):
    ax.plot([ci_lo[i],ci_hi[i]], [i,i], color=colors_ci[i], linewidth=2)
    ax.scatter([deltas_ci[i]], [i], color=colors_ci[i], s=40, zorder=3)
ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
ax.set_yticks(range(len(labels_ci))); ax.set_yticklabels(labels_ci, fontsize=7)
ax.set_xlabel("\u0394AUC (codon\u2212char, pp)", fontsize=8)
ax.set_title("h  DeLong 95% CI", fontsize=10, fontweight="bold", loc="left")

# (i) Char ceiling
ax = axes[2, 2]
char_v = ["v1","v2","v3a","v3b"]
char_mlp = [ABLATION[v]["char_mlp"] for v in char_v]
codon_mlp = [ABLATION[v]["codon_mlp"] for v in char_v]
x = np.arange(len(char_v))
ax.bar(x-0.15, codon_mlp, 0.3, color=CC, label="Codon", edgecolor="gray", linewidth=0.5)
ax.bar(x+0.15, char_mlp, 0.3, color=CH, label="Char", edgecolor="gray", linewidth=0.5)
ax.axhline(y=0.662, color=CH, linestyle="--", linewidth=1, alpha=0.7, label="Char ceiling (0.662)")
ax.set_xticks(x); ax.set_xticklabels(char_v, fontsize=8)
ax.set_ylabel("Task 2 MLP AUC", fontsize=8); ax.legend(fontsize=6)
ax.set_title("i  Char ceiling \u2264 0.662", fontsize=10, fontweight="bold", loc="left")

save_fig(fig, "fig3_ablation.png")