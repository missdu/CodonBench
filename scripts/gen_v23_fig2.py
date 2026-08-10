"""V23 Fig 2: cLM-pLM Crossover & Model Comparison (3x3=9)"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT = ROOT / "paper" / "figures_v2" / "v23"

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

MODELS = {
    "EnCodon-620M":{"type":"cLM","t1_lr":0.617,"t1_std":0.014,"t2_lr":0.785,"t2_std":0.012},
    "CodonBERT":{"type":"cLM","t1_lr":0.660,"t1_std":0.014,"t2_lr":0.734,"t2_std":0.007},
    "CodonTransformer":{"type":"cLM","t1_lr":0.694,"t1_std":0.027,"t2_lr":0.713,"t2_std":0.016},
    "CodonBERT-HF":{"type":"cLM","t1_lr":0.659,"t1_std":0.014,"t2_lr":0.705,"t2_std":0.013},
    "EnCodon-80M":{"type":"cLM","t1_lr":0.633,"t1_std":0.023,"t2_lr":0.686,"t2_std":0.030},
    "CaLM":{"type":"cLM","t1_lr":0.689,"t1_std":0.017,"t2_lr":0.673,"t2_std":0.017},
    "Mistral-Codon-117M":{"type":"cLM","t1_lr":0.611,"t1_std":0.017,"t2_lr":0.656,"t2_std":0.027},
    "cdsBERT-plus":{"type":"cLM","t1_lr":0.623,"t1_std":0.022,"t2_lr":0.601,"t2_std":0.016},
    "cdsBERT":{"type":"cLM","t1_lr":0.629,"t1_std":0.029,"t2_lr":0.598,"t2_std":0.021},
    "Mistral-Codon-16M":{"type":"cLM","t1_lr":0.671,"t1_std":0.025,"t2_lr":0.598,"t2_std":0.012},
    "Mistral-Codon-1M":{"type":"cLM","t1_lr":0.666,"t1_std":0.016,"t2_lr":0.591,"t2_std":0.009},
    "ESM-2-650M":{"type":"pLM","t1_lr":0.719,"t1_std":0.016,"t2_lr":0.680,"t2_std":0.023},
    "ESM-1b":{"type":"pLM","t1_lr":0.711,"t1_std":0.007,"t2_lr":0.600,"t2_std":0.013},
    "NT-v2-500M":{"type":"DNA LM","t1_lr":0.570,"t1_std":0.021,"t2_lr":0.699,"t2_std":0.008},
    "NT-v2-50M":{"type":"DNA LM","t1_lr":0.573,"t1_std":0.018,"t2_lr":0.643,"t2_std":0.019},
    "onehot_pos":{"type":"Traditional","t1_lr":0.755,"t1_std":0.011,"t2_lr":0.891,"t2_std":0.021},
    "kmer6":{"type":"Traditional","t1_lr":0.651,"t1_std":0.025,"t2_lr":0.700,"t2_std":0.029},
    "onehot_freq":{"type":"Traditional","t1_lr":0.667,"t1_std":0.014,"t2_lr":0.589,"t2_std":0.010},
    "combined":{"type":"Traditional","t1_lr":0.666,"t1_std":0.017,"t2_lr":0.594,"t2_std":0.013},
    "kmer4":{"type":"Traditional","t1_lr":0.627,"t1_std":0.019,"t2_lr":0.580,"t2_std":0.024},
    "kmer3":{"type":"Traditional","t1_lr":0.621,"t1_std":0.020,"t2_lr":0.562,"t2_std":0.021},
}
MODEL_ORDER = list(MODELS.keys())

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.45)

# (a) Crossover bar chart
ax = axes[0, 0]
cm = ["EnCodon-620M","CodonBERT","CodonBERT-HF","EnCodon-80M","ESM-2-650M","ESM-1b"]
cl = ["EC-620M","CB","CB-HF","EC-80M","ESM-2","ESM-1b"]
x = np.arange(len(cm)); w = 0.35
ax.bar(x-w/2, [MODELS[m]["t1_lr"] for m in cm], w, label="Missense", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.bar(x+w/2, [MODELS[m]["t2_lr"] for m in cm], w, label="Synonymous", color="#E65100", edgecolor="gray", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(cl, fontsize=7); ax.set_ylabel("AUC (LR)", fontsize=8)
ax.legend(fontsize=7); ax.set_title("a", fontsize=14, fontweight="bold", loc="left")

# (b) Forest: Task 1
ax = axes[0, 1]
esm2_t1 = 0.719
for i, name in enumerate(MODEL_ORDER):
    d = MODELS[name]; delta = (d["t1_lr"]-esm2_t1)*100; ci = 1.96*d["t1_std"]*100
    mtype = d["type"]
    c = "#3182BD" if "cLM" in mtype else ("#C62828" if "pLM" in mtype else ("#E08214" if "DNA" in mtype else "#757575"))
    ax.errorbar(delta, i, xerr=ci, fmt="o", color=c, ecolor=c, elinewidth=1.5, capsize=2, markersize=4)
ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
short_names = [n.replace("Mistral-Codon-","MC-").replace("CodonBERT-HF","CB-HF").replace("CodonBERT","CB").replace("EnCodon-","EC-").replace("NT-v2-","NT-") for n in MODEL_ORDER]
ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(short_names, fontsize=5)
ax.set_xlabel("\u0394AUC vs ESM-2 (pp)", fontsize=8); ax.set_title("b  Task 1 (Missense)", fontsize=10, fontweight="bold", loc="left")

# (c) Forest: Task 2
ax = axes[0, 2]
esm2_t2 = 0.680
for i, name in enumerate(MODEL_ORDER):
    d = MODELS[name]; delta = (d["t2_lr"]-esm2_t2)*100; ci = 1.96*d["t2_std"]*100
    mtype = d["type"]
    c = "#3182BD" if "cLM" in mtype else ("#C62828" if "pLM" in mtype else ("#E08214" if "DNA" in mtype else "#757575"))
    ax.errorbar(delta, i, xerr=ci, fmt="o", color=c, ecolor=c, elinewidth=1.5, capsize=2, markersize=4)
ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
ax.set_yticks(range(len(MODEL_ORDER))); ax.set_yticklabels(short_names, fontsize=5)
ax.set_xlabel("\u0394AUC vs ESM-2 (pp)", fontsize=8); ax.set_title("c  Task 2 (Synonymous)", fontsize=10, fontweight="bold", loc="left")

# (d) Butterfly: codon-level
ax = axes[1, 0]
codon_clms = ["EnCodon-620M","CodonBERT","CodonTransformer","CodonBERT-HF","EnCodon-80M","CaLM"]
codon_labels = ["EC-620M","CB","CT","CB-HF","EC-80M","CaLM"]
y_pos = range(len(codon_clms))
ax.barh([y-0.15 for y in y_pos], [MODELS[m]["t1_lr"] for m in codon_clms], height=0.3, color="#1565C0", label="Missense", edgecolor="gray", linewidth=0.5)
ax.barh([y+0.15 for y in y_pos], [MODELS[m]["t2_lr"] for m in codon_clms], height=0.3, color="#E65100", label="Synonymous", edgecolor="gray", linewidth=0.5)
ax.axvline(x=0.680, color="#C62828", linestyle="--", linewidth=1, alpha=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels(codon_labels, fontsize=7)
ax.set_xlabel("AUC (LR)", fontsize=8); ax.legend(fontsize=6)
ax.set_title("d  Codon-level cLMs", fontsize=10, fontweight="bold", loc="left")

# (e) Butterfly: char/MoE
ax = axes[1, 1]
other_clms = ["cdsBERT-plus","cdsBERT","MC-117M","MC-16M","MC-1M"]
other_keys = ["cdsBERT-plus","cdsBERT","Mistral-Codon-117M","Mistral-Codon-16M","Mistral-Codon-1M"]
y_pos = range(len(other_clms))
ax.barh([y-0.15 for y in y_pos], [MODELS[m]["t1_lr"] for m in other_keys], height=0.3, color="#1565C0", label="Missense", edgecolor="gray", linewidth=0.5)
ax.barh([y+0.15 for y in y_pos], [MODELS[m]["t2_lr"] for m in other_keys], height=0.3, color="#E65100", label="Synonymous", edgecolor="gray", linewidth=0.5)
ax.axvline(x=0.680, color="#C62828", linestyle="--", linewidth=1, alpha=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels(other_clms, fontsize=7)
ax.set_xlabel("AUC (LR)", fontsize=8); ax.legend(fontsize=6)
ax.set_title("e  Char/MoE cLMs", fontsize=10, fontweight="bold", loc="left")

# (f) DNA LM partial recovery
ax = axes[1, 2]
ax.bar(["NT-500M","NT-50M","ESM-2","EC-620M"], [0.699,0.643,0.680,0.785],
       color=["#E08214","#F6C455","#C62828","#08519C"], edgecolor="gray", linewidth=0.5, width=0.5)
ax.axhline(y=0.680, color="#C62828", linestyle="--", linewidth=1, alpha=0.7)
for i, v in enumerate([0.699,0.643,0.680,0.785]): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set_ylabel("Task 2 AUC (LR)", fontsize=8); ax.set_title("f", fontsize=14, fontweight="bold", loc="left")

# (g) CaLM probing depth
ax = axes[2, 0]
ax.bar(["LR","MLP"], [0.673,0.783], color=["#90CAF9","#1565C0"], edgecolor="gray", linewidth=0.5, width=0.4)
ax.axhline(y=0.680, color="#C62828", linestyle="--", linewidth=1.5, label="ESM-2 LR (0.680)")
ax.text(0, 0.683, "0.673", ha="center", fontsize=9, fontweight="bold")
ax.text(1, 0.793, "0.783", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Task 2 AUC", fontsize=8); ax.legend(fontsize=7)
ax.set_title("g  CaLM", fontsize=10, fontweight="bold", loc="left")

# (h) LOGO-CV synonymous
ax = axes[2, 1]
logo_m = ["onehot_pos","CB","CB-HF","EC-80M","onehot_freq"]
logo_std = [0.739,0.518,0.518,0.518,0.518]
logo_logo = [0.739,0.518,0.518,0.518,0.518]
try:
    import json
    with open(ROOT / "results" / "logo_cv_results.json") as f:
        logo_data = json.load(f)
    logo_m = list(logo_data.keys())
    logo_labels_s = [m.replace("codonbert_hf","CB-HF").replace("codonbert","CB").replace("encodon-80m","EC-80M") for m in logo_m]
    logo_std = [logo_data[m].get("standard_cv_auc",0) for m in logo_m]
    logo_logo = [logo_data[m].get("auc_mean",0) for m in logo_m]
except: pass
y_pos = range(len(logo_m))
for i in range(len(logo_m)):
    ax.plot([logo_std[i], logo_logo[i]], [i, i], "o-", color="#3182BD", markersize=5, linewidth=1.5)
    ax.scatter([logo_std[i]], [i], color="#90CAF9", s=40, zorder=3)
    ax.scatter([logo_logo[i]], [i], color="#1565C0", s=40, zorder=3)
ax.set_yticks(y_pos); ax.set_yticklabels([m.replace("codonbert_hf","CB-HF").replace("codonbert","CB").replace("encodon-80m","EC-80M") for m in logo_m], fontsize=7)
ax.set_xlabel("AUC", fontsize=8); ax.set_title("h  LOGO-CV Syn", fontsize=10, fontweight="bold", loc="left")

# (i) LOGO-CV missense
ax = axes[2, 2]
lm = ["onehot_pos","ESM-2","onehot_freq","CodonBERT","EnCodon-80M","CodonBERT-HF"]
ls = [0.755,0.719,0.667,0.660,0.633,0.659]
ll = [0.740,0.640,0.607,0.595,0.588,0.550]
y_pos = range(len(lm))
for i in range(len(lm)):
    ax.plot([ls[i], ll[i]], [i, i], "o-", color="#C62828", markersize=5, linewidth=1.5)
    ax.scatter([ls[i]], [i], color="#EF5350", s=40, zorder=3)
    ax.scatter([ll[i]], [i], color="#C62828", s=40, zorder=3)
ax.set_yticks(y_pos); ax.set_yticklabels(lm, fontsize=7)
ax.set_xlabel("AUC", fontsize=8); ax.set_title("i  LOGO-CV Mis", fontsize=10, fontweight="bold", loc="left")

save_fig(fig, "fig2_crossover_forest.png")