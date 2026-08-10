import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

base = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results\supplementary")
out_dir = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v24")
out_dir.mkdir(parents=True, exist_ok=True)

models_all = [
    ("codonbert", "CodonBERT", "cLM"),
    ("codonbert_hf", "CodonBERT-HF", "cLM"),
    ("encodon-80m", "EnCodon-80M", "cLM"),
    ("esm2", "ESM-2-650M", "pLM"),
]
tasks = [
    ("task3_synonymous", "SynPath"),
    ("task2_missense", "MisPath"),
]

fig, axes = plt.subplots(4, 2, figsize=(9, 16))

for i, (model_key, model_label, model_type) in enumerate(models_all):
    for j, (task_key, task_label) in enumerate(tasks):
        ax = axes[i, j]
        
        if model_key == "esm2":
            tsne_file = base / f"esm2_{task_key}_tsne.npy"
            label_file = base / f"esm2_{task_key}_tsne_labels.npy"
        else:
            tsne_file = base / f"{model_key}_{task_key}_tsne.npy"
            label_file = base / f"{model_key}_{task_key}_tsne_labels.npy"
        
        tsne = np.load(tsne_file)
        labels = np.load(label_file)
        
        benign = labels == 0
        pathogenic = labels == 1
        
        ax.scatter(tsne[benign, 0], tsne[benign, 1],
                   c='#4A90D9', alpha=0.3, s=8, label='Benign', rasterized=True)
        ax.scatter(tsne[pathogenic, 0], tsne[pathogenic, 1],
                   c='#E8833A', alpha=0.5, s=12, label='Pathogenic', rasterized=True)
        
        ax.set_title(f"{model_label} — {task_label}", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0 and j == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

for i, (_, mlabel, mtype) in enumerate(models_all):
    axes[i, 0].annotate(mtype, xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', ha='center', va='center', rotation=90,
                        color='#2B5F8A' if mtype == "cLM" else '#E8833A')

plt.tight_layout()
plt.savefig(out_dir / "fig_tsne_clm_vs_plm.png", dpi=300, bbox_inches='tight')
plt.savefig(out_dir / "fig_tsne_clm_vs_plm.pdf", bbox_inches='tight')
plt.close()
print("Saved fig_tsne_clm_vs_plm.png/pdf")

# SynPath-only horizontal panel
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
for i, (model_key, model_label, model_type) in enumerate(models_all):
    ax = axes2[i]
    
    if model_key == "esm2":
        tsne_file = base / f"esm2_task3_synonymous_tsne.npy"
        label_file = base / f"esm2_task3_synonymous_tsne_labels.npy"
    else:
        tsne_file = base / f"{model_key}_task3_synonymous_tsne.npy"
        label_file = base / f"{model_key}_task3_synonymous_tsne_labels.npy"
    
    tsne = np.load(tsne_file)
    labels = np.load(label_file)
    
    benign = labels == 0
    pathogenic = labels == 1
    
    ax.scatter(tsne[benign, 0], tsne[benign, 1],
               c='#4A90D9', alpha=0.3, s=8, label='Benign', rasterized=True)
    ax.scatter(tsne[pathogenic, 0], tsne[pathogenic, 1],
               c='#E8833A', alpha=0.5, s=12, label='Pathogenic', rasterized=True)
    
    ax.set_title(f"{model_label} ({model_type})", fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

axes2[0].legend(fontsize=9, loc='upper right', framealpha=0.8)
fig2.suptitle("SynPath: t-SNE of embeddings", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out_dir / "fig_tsne_synpath_clm_vs_plm.png", dpi=300, bbox_inches='tight')
plt.savefig(out_dir / "fig_tsne_synpath_clm_vs_plm.pdf", bbox_inches='tight')
plt.close()
print("Saved fig_tsne_synpath_clm_vs_plm.png/pdf")