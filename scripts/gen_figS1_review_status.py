import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench")
OUT_DIR = BASE / "paper" / "figures_v2" / "v26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE / "results" / "review_status_stratified.json", "r", encoding="utf-16") as f:
    rev_data = json.load(f)

tiers = {4: {"n": 200, "auc": 0.725, "std": 0.117},
         3: {"n": 2640, "auc": 0.591, "std": 0.018}}

fig, ax = plt.subplots(figsize=(6, 5))

labels = [f"Tier 4 (expert panel)\nn = {tiers[4]['n']}", f"Tier 3 (single submitter)\nn = {tiers[3]['n']}"]
aucs = [tiers[4]["auc"], tiers[3]["auc"]]
stds = [tiers[4]["std"], tiers[3]["std"]]

bars = ax.bar(labels, aucs, yerr=stds, capsize=8, color=["#2166ac", "#92c5de"],
              edgecolor="black", linewidth=0.5, width=0.5)

for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{auc:.3f}", ha="center", fontsize=11, fontweight="bold")

ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_ylabel("SynPath AUC (CodonBERT-HF)", fontsize=11)
ax.set_title("Supplementary Figure S1: Review status stratification", fontsize=12, fontweight="bold")
ax.set_ylim(0.3, 0.95)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT_DIR / "figS1_review_status.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "figS1_review_status.pdf", bbox_inches="tight")
print("Saved figS1_review_status.png and .pdf")