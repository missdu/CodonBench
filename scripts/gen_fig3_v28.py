import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "lr_mlp_full_comparison.json", "r", encoding="utf-8") as f:
    full_data = json.load(f)
with open(BASE / "results" / "layerwise_analysis_results.json", "r", encoding="utf-8") as f:
    layer_data = json.load(f)

CODON_MODELS = ["encodon-80m", "encodon-620m", "codonbert", "codonbert_hf", "codontransformer",
                "calm", "mistralcodon117m", "mistralcodon16m", "mistralcodon1m"]
CHAR_MODELS = ["cdsbert"]
PLM_MODELS = ["esm2_650m", "esm1b"]
ALL_CDS = CODON_MODELS + CHAR_MODELS

def get_color(m):
    t = tok_dict.get(m, "codon")
    if t == "codon": return CODON_C
    if t == "char": return CHAR_C
    return PLM_C

CODON_LABELS = {"encodon-80m": "EnCodon-80M", "encodon-620m": "EnCodon-620M",
                "codonbert": "CodonBERT", "codonbert_hf": "CodonBERT-HF",
                "codontransformer": "CodonTrans.", "calm": "CaLM",
                "mistralcodon117m": "MC-117M", "mistralcodon16m": "MC-16M",
                "mistralcodon1m": "MC-1M", "cdsbert": "cdsBERT",
                "esm2_650m": "ESM-2", "esm1b": "ESM-1b"}

lr_dict = {}
mlp_dict = {}
tok_dict = {}
for entry in full_data:
    lr_dict[(entry["model"], entry["task"])] = entry["lr_auc"]
    mlp_dict[(entry["model"], entry["task"])] = entry["mlp_auc"]
    tok_dict[entry["model"]] = entry["tokenization"]

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 4, hspace=0.40, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

CODON_C = "#2166ac"
CHAR_C = "#d6604d"
PLM_C = "#2ca02c"

# --- Panel a: LR-to-MLP slope chart, SynPath ---
ax_a = fig.add_subplot(gs[0, 0])
models_syn = []
for m in ALL_CDS:
    lr = lr_dict.get((m, "task3_synonymous"))
    mlp_v = mlp_dict.get((m, "task3_synonymous"))
    if lr is not None and mlp_v is not None:
        models_syn.append((m, lr, mlp_v))
for m in PLM_MODELS:
    lr = lr_dict.get((m, "task3_synonymous"))
    mlp_v = mlp_dict.get((m, "task3_synonymous"))
    if lr is not None and mlp_v is not None:
        models_syn.append((m, lr, mlp_v))

for i, (m, lr, mlp_v) in enumerate(models_syn):
    c = get_color(m)
    ax_a.plot([0, 1], [lr, mlp_v], "o-", color=c, markersize=5, linewidth=1.2, alpha=0.8)
ax_a.set_xlim(-0.2, 1.2)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(["LR", "MLP"], fontsize=9)
ax_a.set_ylabel("AUC", fontsize=10)
ax_a.set_title("a  SynPath", fontsize=11, fontweight="bold", loc="left")
ax_a.grid(True, alpha=0.3)
ax_a.set_ylim(0.45, 0.95)

# --- Panel b: LR-to-MLP slope chart, MisPath ---
ax_b = fig.add_subplot(gs[0, 1])
models_mis = []
for m in ALL_CDS:
    lr = lr_dict.get((m, "task2_missense"))
    mlp_v = mlp_dict.get((m, "task2_missense"))
    if lr is not None and mlp_v is not None:
        models_mis.append((m, lr, mlp_v))
for m in PLM_MODELS:
    lr = lr_dict.get((m, "task2_missense"))
    mlp_v = mlp_dict.get((m, "task2_missense"))
    if lr is not None and mlp_v is not None:
        models_mis.append((m, lr, mlp_v))

for i, (m, lr, mlp_v) in enumerate(models_mis):
    c = get_color(m)
    ax_b.plot([0, 1], [lr, mlp_v], "o-", color=c, markersize=5, linewidth=1.2, alpha=0.8)
ax_b.set_xlim(-0.2, 1.2)
ax_b.set_xticks([0, 1])
ax_b.set_xticklabels(["LR", "MLP"], fontsize=9)
ax_b.set_ylabel("AUC", fontsize=10)
ax_b.set_title("b  MisPath", fontsize=11, fontweight="bold", loc="left")
ax_b.grid(True, alpha=0.3)
ax_b.set_ylim(0.45, 0.95)

# --- Panel c: Protocol sensitivity scatter ---
ax_c = fig.add_subplot(gs[0, 2])
syn_lr, syn_mlp = [], []
mis_lr, mis_mlp = [], []
for m in ALL_CDS + PLM_MODELS:
    for task, lr_l, mlp_l in [("task3_synonymous", syn_lr, syn_mlp),
                               ("task2_missense", mis_lr, mis_mlp)]:
        lr = lr_dict.get((m, task))
        ml = mlp_dict.get((m, task))
        if lr is not None and ml is not None:
            lr_l.append(lr)
            mlp_l.append(ml)

ax_c.scatter(mis_lr, mis_mlp, c="#4393c3", marker="o", s=60, zorder=5, label="MisPath")
ax_c.scatter(syn_lr, syn_mlp, c="#d6604d", marker="D", s=60, zorder=5, label="SynPath")
lims = [0.45, 0.95]
ax_c.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
if len(syn_lr) >= 3:
    rho_syn, p_syn = stats.spearmanr(syn_lr, syn_mlp)
    ax_c.text(0.05, 0.95, f"SynPath: ρ={rho_syn:.3f}\np={p_syn:.3f}",
              transform=ax_c.transAxes, fontsize=8, color="#d6604d", va="top")
if len(mis_lr) >= 3:
    rho_mis, p_mis = stats.spearmanr(mis_lr, mis_mlp)
    ax_c.text(0.05, 0.78, f"MisPath: ρ={rho_mis:.3f}\np={p_mis:.3f}",
              transform=ax_c.transAxes, fontsize=8, color="#4393c3", va="top")
ax_c.set_xlabel("LR AUC", fontsize=10)
ax_c.set_ylabel("MLP AUC", fontsize=10)
ax_c.set_title("c  Protocol sensitivity", fontsize=11, fontweight="bold", loc="left")
ax_c.legend(fontsize=8, loc="lower right")
ax_c.set_xlim(lims)
ax_c.set_ylim(lims)

ax_c.grid(True, alpha=0.3)

# --- Panel d: Loss insensitivity (LLR vs MLP gap) ---
ax_d = fig.add_subplot(gs[0, 3])
llr_models = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M"]
llr_syn = [0.526, 0.484, 0.495]
llr_mis = [0.491, 0.763, 0.656]
mlp_syn = [0.849, 0.816, 0.812]
mlp_mis = [0.677, 0.650, 0.674]

x = np.arange(len(llr_models))
w = 0.18
ax_d.bar(x - 1.5*w, llr_syn, w, label="LLR SynPath", color="#d6604d", alpha=0.6)
ax_d.bar(x - 0.5*w, mlp_syn, w, label="MLP SynPath", color="#d6604d", alpha=0.9)
ax_d.bar(x + 0.5*w, llr_mis, w, label="LLR MisPath", color="#4393c3", alpha=0.6)
ax_d.bar(x + 1.5*w, mlp_mis, w, label="MLP MisPath", color="#4393c3", alpha=0.9)
ax_d.set_xticks(x)
ax_d.set_xticklabels(llr_models, fontsize=8, rotation=20, ha="right")
ax_d.set_ylabel("AUC", fontsize=10)
ax_d.set_title("d  Loss insensitivity", fontsize=11, fontweight="bold", loc="left")
ax_d.legend(fontsize=7, loc="upper right")
ax_d.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
ax_d.grid(True, alpha=0.3, axis="y")
ax_d.set_ylim(0.45, 0.95)

# --- Panel e: Layer-wise SynPath ---
ax_e = fig.add_subplot(gs[1, 0:2])
syn_layers = [(x["layer"], x["lr_auc"], x["mlp_auc"])
              for x in layer_data
              if x["model"] == "CodonBERT-HF" and x["task"] == "task3_synonymous"
              and isinstance(x["layer"], int)]
syn_layers.sort(key=lambda x: x[0])
layers = [d[0] for d in syn_layers]
lr_v = [d[1] for d in syn_layers]
mlp_v = [d[2] for d in syn_layers]
ax_e.plot(layers, lr_v, "o-", color="#2166ac", label="LR", linewidth=1.5, markersize=4)
ax_e.plot(layers, mlp_v, "s-", color="#d6604d", label="MLP", linewidth=1.5, markersize=4)
ax_e.fill_between(layers, lr_v, mlp_v, alpha=0.12, color="gray")
ax_e.set_xlabel("Layer", fontsize=10)
ax_e.set_ylabel("AUC", fontsize=10)
ax_e.set_title("e  Layer-wise (CodonBERT-HF, SynPath)", fontsize=11, fontweight="bold", loc="left")
ax_e.legend(fontsize=9)
ax_e.grid(True, alpha=0.3)
ax_e.set_ylim(0.55, 0.90)

# --- Panel f: Layer-wise MisPath ---
ax_f = fig.add_subplot(gs[1, 2:4])
mis_layers = [(x["layer"], x["lr_auc"], x["mlp_auc"])
              for x in layer_data
              if x["model"] == "CodonBERT-HF" and x["task"] == "task2_missense"
              and isinstance(x["layer"], int)]
mis_layers.sort(key=lambda x: x[0])
layers_m = [d[0] for d in mis_layers]
lr_m = [d[1] for d in mis_layers]
mlp_m = [d[2] for d in mis_layers]
ax_f.plot(layers_m, lr_m, "o-", color="#2166ac", label="LR", linewidth=1.5, markersize=4)
ax_f.plot(layers_m, mlp_m, "s-", color="#d6604d", label="MLP", linewidth=1.5, markersize=4)
ax_f.fill_between(layers_m, lr_m, mlp_m, alpha=0.12, color="gray")
ax_f.set_xlabel("Layer", fontsize=10)
ax_f.set_ylabel("AUC", fontsize=10)
ax_f.set_title("f  Layer-wise (CodonBERT-HF, MisPath)", fontsize=11, fontweight="bold", loc="left")
ax_f.legend(fontsize=9)
ax_f.grid(True, alpha=0.3)
ax_f.set_ylim(0.55, 0.90)

fig.suptitle("Figure 3 | The probing-depth confound is channel-specific",
             fontsize=13, fontweight="bold", y=0.98)

plt.savefig(OUT_DIR / "fig3_combined.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig3_combined.pdf", bbox_inches="tight")
print("Saved fig3_combined.png and .pdf")