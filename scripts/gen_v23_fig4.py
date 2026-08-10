"""V23 Fig 4: Probing Depth & Fine-Tuning (2x3=6)"""
import numpy as np, matplotlib.pyplot as plt, json
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

# (a) Production-model LR→MLP slope chart (SynPath)
ax = axes[0, 0]
models_slope = {
    "EnCodon-620M":     {"lr": 0.785, "mlp": 0.800, "type": "cLM-codon"},
    "CodonBERT":        {"lr": 0.734, "mlp": 0.849, "type": "cLM-codon"},
    "CodonTransformer": {"lr": 0.713, "mlp": 0.832, "type": "cLM-codon"},
    "CodonBERT-HF":     {"lr": 0.705, "mlp": 0.816, "type": "cLM-codon"},
    "EnCodon-80M":      {"lr": 0.686, "mlp": 0.812, "type": "cLM-codon"},
    "CaLM":             {"lr": 0.673, "mlp": 0.783, "type": "cLM-contrastive"},
    "Mistral-16M":      {"lr": 0.598, "mlp": 0.841, "type": "cLM-MoE"},
    "cdsBERT":          {"lr": 0.598, "mlp": 0.627, "type": "cLM-char"},
    "ESM-2":            {"lr": 0.680, "mlp": 0.680, "type": "pLM"},
}
type_colors = {"cLM-codon": "#2166AC", "cLM-contrastive": "#7B3294", "cLM-MoE": "#D95F02", "cLM-char": "#E08214", "pLM": "#1B7837"}
for i, (name, d) in enumerate(models_slope.items()):
    c = type_colors[d["type"]]
    ax.plot([0, 1], [d["lr"], d["mlp"]], "o-", color=c, linewidth=1.5, markersize=5, alpha=0.85)
    y_label = d["mlp"] if d["mlp"] > d["lr"] else d["lr"]
    ax.text(1.02, d["mlp"], name, fontsize=5.5, va="center", color=c)
ax.set_xticks([0, 1]); ax.set_xticklabels(["LR", "MLP"], fontsize=9)
ax.set_ylabel("SynPath AUC", fontsize=8)
ax.set_xlim(-0.15, 1.35); ax.set_ylim(0.55, 0.90)
ax.axhline(y=0.680, color="#1B7837", linestyle="--", linewidth=1, alpha=0.5, label="ESM-2 LR (0.680)")
ax.set_title("a  Production models: LR\u2192MLP", fontsize=10, fontweight="bold", loc="left")

# (b) Probing ablation: CB-HF SynPath
ax = axes[0, 1]
with open(ROOT / "results" / "probing_ablation_codonbert_hf_task3.json") as f:
    probe_cbhf = json.load(f)
probes2 = list(probe_cbhf.keys())
aucs2 = [probe_cbhf[p]["mean"] for p in probes2]
stds2 = [probe_cbhf[p]["std"] for p in probes2]
pc2 = ["#90CAF9" if p != "MLP" else "#1565C0" for p in probes2]
ax.bar(probes2, aucs2, yerr=stds2, color=pc2, edgecolor="gray", linewidth=0.5, capsize=3)
ax.set_ylabel("AUC", fontsize=8); ax.set_title("b  CB-HF SynPath", fontsize=10, fontweight="bold", loc="left")

# (c) LR->MLP->LoRA: Task 2
ax = axes[0, 2]
lm = ["CodonBERT","CB-HF","EC-80M","EC-620M","CodonTrans"]
lr_v = [0.734,0.705,0.686,0.785,0.713]; mlp_v = [0.849,0.816,0.812,0.800,0.832]; lora_v = [0.912,0.880,0.893,0.870,0.943]
x = np.arange(len(lm)); w = 0.25
ax.bar(x-w, lr_v, w, label="LR", color="#90CAF9", edgecolor="gray", linewidth=0.5)
ax.bar(x, mlp_v, w, label="MLP", color="#42A5F5", edgecolor="gray", linewidth=0.5)
ax.bar(x+w, lora_v, w, label="LoRA", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(lm, fontsize=7); ax.set_ylabel("AUC", fontsize=8)
ax.legend(fontsize=7); ax.set_title("c  Task 2 (Synonymous)", fontsize=10, fontweight="bold", loc="left")

# (d) LR->MLP->LoRA: Task 1
ax = axes[1, 0]
lr_t1 = [0.660,0.659,0.633,0.617,0.694]; mlp_t1 = [0.677,0.650,0.674,0.632,0.706]; lora_t1 = [0.808,0.814,0.696,0.726,0.813]
ax.bar(x-w, lr_t1, w, label="LR", color="#90CAF9", edgecolor="gray", linewidth=0.5)
ax.bar(x, mlp_t1, w, label="MLP", color="#42A5F5", edgecolor="gray", linewidth=0.5)
ax.bar(x+w, lora_t1, w, label="LoRA", color="#1565C0", edgecolor="gray", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(lm, fontsize=7); ax.set_ylabel("AUC", fontsize=8)
ax.legend(fontsize=7); ax.set_title("d  Task 1 (Missense)", fontsize=10, fontweight="bold", loc="left")

# (e) Oracle ceiling
ax = axes[1, 1]
om = ["Oracle\n(CV)","CodonTrans\nLoRA","CB\nLoRA","EC-80M\nLoRA","CB-HF\nLoRA","EC-620M\nLoRA"]
oa = [0.849,0.943,0.912,0.893,0.880,0.870]
oc = ["#4CAF50","#1565C0","#3182BD","#9ECAE1","#6BAED6","#08519C"]
ax.bar(om, oa, color=oc, edgecolor="gray", linewidth=0.5)
ax.axhline(y=0.949, color="#4CAF50", linestyle="--", linewidth=1.5, label="Oracle test (0.949)")
for i, v in enumerate(oa): ax.text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")
ax.set_ylabel("AUC", fontsize=8); ax.legend(fontsize=6)
ax.set_title("e  Oracle ceiling", fontsize=10, fontweight="bold", loc="left")

# (f) LoRA rank ablation
ax = axes[1, 2]
ranks = [4,8,16,32]
cb_r = [0.908,0.912,0.918,0.920]; cbhf_r = [0.872,0.880,0.886,0.890]; ec_r = [0.885,0.893,0.898,0.901]
ax.plot(ranks, cb_r, "o-", color="#3182BD", label="CB", linewidth=2, markersize=6)
ax.plot(ranks, cbhf_r, "o-", color="#6BAED6", label="CB-HF", linewidth=2, markersize=6)
ax.plot(ranks, ec_r, "o-", color="#9ECAE1", label="EC-80M", linewidth=2, markersize=6)
ax.set_xlabel("LoRA rank (r)", fontsize=8); ax.set_ylabel("Task 2 AUC", fontsize=8)
ax.legend(fontsize=7); ax.set_title("f  LoRA rank ablation", fontsize=10, fontweight="bold", loc="left")

save_fig(fig, "fig4_probing_lora.png")