"""V23 Fig 1: Framework & Information Decomposition (3x3=9)"""
import json, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT = ROOT / "paper" / "figures_v2" / "v23"
OUT.mkdir(exist_ok=True, parents=True)

def save_fig(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
plt.subplots_adjust(wspace=0.35, hspace=0.4)

# (a) Information decomposition
ax = axes[0, 0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.add_patch(mpatches.FancyBboxPatch((0.5, 3), 3, 4, boxstyle="round,pad=0.3", facecolor="#90CAF9", edgecolor="#1565C0", linewidth=2))
ax.text(2, 5.8, "I(A;Y)", ha="center", va="center", fontsize=12, fontweight="bold", color="#1565C0")
ax.text(2, 4.2, "AA channel", ha="center", va="center", fontsize=9, color="#1565C0")
ax.add_patch(mpatches.FancyBboxPatch((6.5, 3), 3, 4, boxstyle="round,pad=0.3", facecolor="#FFCC80", edgecolor="#E65100", linewidth=2))
ax.text(8, 5.8, "I(\u03c3;Y|A)", ha="center", va="center", fontsize=12, fontweight="bold", color="#E65100")
ax.text(8, 4.2, "Synonymous channel", ha="center", va="center", fontsize=9, color="#E65100")
ax.annotate("", xy=(5.8, 5), xytext=(4.2, 5), arrowprops=dict(arrowstyle="<->", color="black", lw=2))
ax.text(5, 7, "I(CDS;Y) = I(A;Y) + I(\u03c3;Y|A)", ha="center", va="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
ax.set_title("a", fontsize=14, fontweight="bold", loc="left"); ax.axis("off")

# (b) Tokenization × probing-depth grid (VISUAL CENTER - moved from g)
ax = axes[0, 1]
signal_strength = np.array([[0.3,0.8,1.0],[0.2,0.2,0.2],[0.1,0.1,0.1]])
im = ax.imshow(signal_strength, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["LR","MLP","LoRA"], fontsize=9, fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels(["Codon","Character","AA-only"], fontsize=9, fontweight="bold")
for i in range(3):
    for j in range(3):
        val = signal_strength[i,j]
        label_map = {(0,0):"n.s.",(0,1):"+9\u201314pp***",(0,2):">0.87",
                     (1,0):"n.s.",(1,1):"\u22640.66",(1,2):"N/A",
                     (2,0):"n.a.",(2,1):"n.a.",(2,2):"n.a."}
        txt = label_map.get((i,j), f"{val:.1f}")
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if val>0.5 else "black")
ax.set_title("b  Tokenization \u00d7 probing depth", fontsize=11, fontweight="bold", loc="left")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Extractable signal", fontsize=7)

# (c) Task-channel mapping
ax = axes[0, 2]
tasks = ["T1\nMissense", "T2\nSynonymous", "T3\nmRFP", "T4\nE.coli", "T5\nmRNA", "T6\nFungal"]
n_samples = [5000, 2840, 1459, 3000, 5000, 7089]
channel = ["I(A;Y)", "I(\u03c3;Y|A)", "I(\u03c3;Y|A)", "I(A;Y)", "Dual", "I(\u03c3;Y|A)"]
ch_colors = ["#1565C0", "#E65100", "#E65100", "#1565C0", "#7B1FA2", "#E65100"]
ax.bar(range(6), n_samples, color=ch_colors, edgecolor="gray", linewidth=0.5)
for i, (c) in enumerate(channel):
    ax.text(i, n_samples[i]+150, c, ha="center", va="bottom", fontsize=6, fontweight="bold", color=ch_colors[i])
ax.set_xticks(range(6)); ax.set_xticklabels(tasks, fontsize=7)
ax.set_ylabel("Samples", fontsize=8); ax.set_title("c", fontsize=14, fontweight="bold", loc="left")

# (d) Four-level protocol
ax = axes[1, 0]
levels = ["Level 0\nZero-shot", "Level 1\nLR", "Level 2\nMLP", "Level 3\nLoRA"]
widths = [0.5, 1.0, 1.5, 2.0]; level_colors = ["#E0E0E0", "#90CAF9", "#42A5F5", "#1565C0"]
for i, (lv, w, c) in enumerate(zip(levels, widths, level_colors)):
    ax.barh(i, w, height=0.6, color=c, edgecolor="gray", linewidth=0.5)
    ax.text(w+0.05, i, lv, ha="left", va="center", fontsize=7)
ax.set_xlim(0, 4); ax.set_yticks([]); ax.set_xlabel("Information extracted \u2192", fontsize=8)
ax.invert_yaxis(); ax.set_title("d", fontsize=14, fontweight="bold", loc="left")

# (e) 21-model landscape
ax = axes[1, 1]
MODEL_ORDER = ["EnCodon-620M","CodonBERT","CodonTransformer","CodonBERT-HF","EnCodon-80M",
    "CaLM","Mistral-Codon-117M","cdsBERT-plus","cdsBERT","Mistral-Codon-16M","Mistral-Codon-1M",
    "ESM-2-650M","ESM-1b","NT-v2-500M","NT-v2-50M","onehot_pos","kmer6","onehot_freq","combined","kmer4","kmer3"]
MODELS_21 = {
    "EnCodon-620M":{"type":"cLM","params_M":620,"t2_lr":0.785},
    "CodonBERT":{"type":"cLM","params_M":110,"t2_lr":0.734},
    "CodonTransformer":{"type":"cLM","params_M":110,"t2_lr":0.713},
    "CodonBERT-HF":{"type":"cLM","params_M":110,"t2_lr":0.705},
    "EnCodon-80M":{"type":"cLM","params_M":80,"t2_lr":0.686},
    "CaLM":{"type":"cLM","params_M":86,"t2_lr":0.673},
    "Mistral-Codon-117M":{"type":"cLM","params_M":117,"t2_lr":0.656},
    "cdsBERT-plus":{"type":"cLM","params_M":110,"t2_lr":0.601},
    "cdsBERT":{"type":"cLM","params_M":110,"t2_lr":0.598},
    "Mistral-Codon-16M":{"type":"cLM","params_M":16,"t2_lr":0.598},
    "Mistral-Codon-1M":{"type":"cLM","params_M":1,"t2_lr":0.591},
    "ESM-2-650M":{"type":"pLM","params_M":650,"t2_lr":0.680},
    "ESM-1b":{"type":"pLM","params_M":650,"t2_lr":0.600},
    "NT-v2-500M":{"type":"DNA LM","params_M":500,"t2_lr":0.699},
    "NT-v2-50M":{"type":"DNA LM","params_M":50,"t2_lr":0.643},
    "onehot_pos":{"type":"Traditional","params_M":0,"t2_lr":0.891},
    "kmer6":{"type":"Traditional","params_M":0,"t2_lr":0.700},
    "onehot_freq":{"type":"Traditional","params_M":0,"t2_lr":0.589},
    "combined":{"type":"Traditional","params_M":0,"t2_lr":0.594},
    "kmer4":{"type":"Traditional","params_M":0,"t2_lr":0.580},
    "kmer3":{"type":"Traditional","params_M":0,"t2_lr":0.562},
}
MCOLORS = {"EnCodon-620M":"#08519C","CodonBERT":"#3182BD","CodonTransformer":"#1B7837","CodonBERT-HF":"#6BAED6",
    "EnCodon-80M":"#9ECAE1","CaLM":"#A6CEE3","Mistral-Codon-117M":"#F4A582","cdsBERT-plus":"#7FBF7B",
    "cdsBERT":"#B2DF8A","Mistral-Codon-16M":"#FDB863","Mistral-Codon-1M":"#FFE08A",
    "ESM-2-650M":"#C62828","ESM-1b":"#EF5350","NT-v2-500M":"#E08214","NT-v2-50M":"#F6C455",
    "onehot_pos":"#757575","kmer6":"#9E9E9E","onehot_freq":"#BDBDBD","combined":"#A1887F","kmer4":"#D7CCC8","kmer3":"#ECEFF1"}
for name in MODEL_ORDER:
    d = MODELS_21.get(name, {}); params = d.get("params_M", 0); t2 = d.get("t2_lr", 0)
    mtype = d.get("type", "")
    if params > 0 and t2 > 0:
        marker = "o" if "cLM" in mtype else ("s" if "pLM" in mtype else ("^" if "DNA" in mtype else "D"))
        ax.scatter(params, t2, c=MCOLORS.get(name, "#999"), s=60, marker=marker, edgecolors="black", linewidths=0.5, zorder=3)
        short = name.replace("CodonBERT-HF","CB-HF").replace("CodonBERT","CB").replace("EnCodon-","EC-").replace("Mistral-Codon-","MC-").replace("NT-v2-","NT-")
        ax.annotate(short, (params, t2), fontsize=5, ha="left", va="bottom", xytext=(3,3), textcoords="offset points")
ax.axhline(y=0.680, color="#C62828", linestyle="--", linewidth=1, alpha=0.7, label="ESM-2 (0.680)")
ax.set_xscale("log"); ax.set_xlabel("Parameters (M)", fontsize=8); ax.set_ylabel("Task 2 AUC (LR)", fontsize=8)
ax.legend(fontsize=6, loc="lower right"); ax.set_title("e", fontsize=14, fontweight="bold", loc="left")

# (f) Accessibility audit
ax = axes[1, 2]
ax.bar(["Loaded\ndirectly","Adapted","Inaccessible"], [3,7,10], color=["#4CAF50","#FF9800","#F44336"], edgecolor="gray", linewidth=0.5)
for i, c in enumerate([3,7,10]): ax.text(i, c+0.3, str(c), ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("cLMs", fontsize=8); ax.set_title("f", fontsize=14, fontweight="bold", loc="left")

# (g) Codon table: synonymous group sizes (moved from b)
ax = axes[2, 0]
codon_table = {"Phe":2,"Leu":6,"Ile":3,"Met":1,"Val":4,"Ser":6,"Pro":4,"Thr":4,"Ala":4,
               "Tyr":2,"His":2,"Gln":2,"Asn":2,"Lys":2,"Asp":2,"Glu":2,"Cys":2,"Trp":1,"Arg":6,"Gly":4,"Stop":3}
aa_names = list(codon_table.keys()); syn_sizes = list(codon_table.values())
colors_bt = plt.cm.YlOrRd(np.array(syn_sizes) / max(syn_sizes))
ax.barh(range(len(aa_names)), syn_sizes, color=colors_bt, edgecolor="gray", linewidth=0.5)
ax.set_yticks(range(len(aa_names))); ax.set_yticklabels(aa_names, fontsize=6)
ax.set_xlabel("Synonymous codons", fontsize=8); ax.invert_yaxis()
ax.set_title("g", fontsize=14, fontweight="bold", loc="left")

# (h) AlphaMissense
ax = axes[2, 1]
ax.bar(["Missense","Synonymous"], [0.956,0.440], color=["#1565C0","#E65100"], edgecolor="gray", linewidth=0.5, width=0.5)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(0, 0.97, "0.956", ha="center", fontsize=9, fontweight="bold")
ax.text(1, 0.46, "0.440", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("AUC", fontsize=8); ax.set_ylim(0, 1.05); ax.set_title("h", fontsize=14, fontweight="bold", loc="left")

# (i) kmer6 model-free evidence
ax = axes[2, 2]
ax.bar(["kmer6","kmer3","kmer4","onehot_freq"], [0.700,0.562,0.580,0.589],
       color=["#4CAF50","#BDBDBD","#BDBDBD","#BDBDBD"], edgecolor="gray", linewidth=0.5)
ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(y=0.680, color="#C62828", linestyle="--", linewidth=1, alpha=0.7)
ax.text(0, 0.71, "0.700***", ha="center", fontsize=8, fontweight="bold", color="#4CAF50")
ax.set_ylabel("AUC", fontsize=8); ax.set_ylim(0.4, 0.8); ax.set_title("i", fontsize=14, fontweight="bold", loc="left")

save_fig(fig, "fig1_framework.png")