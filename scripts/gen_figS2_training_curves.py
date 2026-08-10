import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
TRAINER_DIR = BASE / "results" / "trainer_states"
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("codon_v1.json",  "v1 codon",    "#2166ac"),
    ("char_v1.json",   "v1 char",     "#d6604d"),
    ("codon_v2.json",  "v2 codon",    "#4393c3"),
    ("char_v2.json",   "v2 char",     "#f4a582"),
    ("codon_v3a.json", "v3a codon",   "#92c5de"),
    ("char_v3a.json",  "v3a char",    "#fddbc7"),
    ("codon_v3b.json", "v3b codon",   "#053061"),
    ("char_v3b.json",  "v3b char",    "#67001f"),
    ("codon_v4.json",  "v4 codon",    "#b2182b"),
    ("char_v4.json",   "v4 char",     "#2166ac"),
]

CONDITIONS = [
    ("v1",  "4.8K CDS, 10 ep\n(~20M params)", ["codon_v1.json", "char_v1.json"]),
    ("v2",  "55K CDS (synth.), 30 ep\n(~20M params)", ["codon_v2.json", "char_v2.json"]),
    ("v3a", "4.8K CDS, 100 ep\n(~20M params)", ["codon_v3a.json", "char_v3a.json"]),
    ("v3b", "114K CDS, 30 ep\n(~20M params)", ["codon_v3b.json", "char_v3b.json"]),
    ("v4",  "114K CDS, 30 ep\n(~110M params)", ["codon_v4.json", "char_v4.json"]),
]

def load_loss(fname):
    with open(TRAINER_DIR / fname, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps, losses = [], []
    for entry in data.get("log_history", []):
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    return np.array(steps), np.array(losses)

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 5, hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97, top=0.92, bottom=0.08)

CODON_COLOR = "#2166ac"
CHAR_COLOR = "#d6604d"

for col, (cond_name, cond_label, fnames) in enumerate(CONDITIONS):
    for row, (fname, tok_type, color) in enumerate([
        (fnames[0], "codon", CODON_COLOR),
        (fnames[1], "char",  CHAR_COLOR),
    ]):
        ax = fig.add_subplot(gs[row, col])
        steps, losses = load_loss(fname)
        ax.plot(steps, losses, color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(f"{cond_name} {tok_type}", fontsize=10, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Training steps", fontsize=9)
        if col == 0:
            ax.set_ylabel("MLM loss", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if len(losses) > 0:
            ax.set_ylim(bottom=min(losses) * 0.95, top=max(losses) * 1.05)

fig.suptitle("Supplementary Figure S2: Pretraining loss curves for from-scratch ablation models",
             fontsize=12, fontweight="bold", y=0.98)

plt.savefig(OUT_DIR / "figS2_training_curves.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS2_training_curves.pdf", bbox_inches="tight")
print("Saved figS2_training_curves.png and .pdf")