import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

base = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results\supplementary")
out_dir = Path(r"F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v24")
out_dir.mkdir(parents=True, exist_ok=True)

models = ["codonbert", "codonbert_hf", "encodon-80m"]
model_labels = ["CodonBERT", "CodonBERT-HF", "EnCodon-80M"]
tasks = [
    ("task3_synonymous", "SynPath"),
    ("task2_missense", "MisPath"),
]

fig, axes = plt.subplots(3, 2, figsize=(10, 14))

for i, (model, mlabel) in enumerate(zip(models, model_labels)):
    for j, (task_key, task_label) in enumerate(tasks):
        ax = axes[i, j]
        tsne_file = base / f"{model}_{task_key}_tsne.npy"
        label_file = base / f"{model}_{task_key}_tsne_labels.npy"
        
        if not tsne_file.exists() or not label_file.exists():
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=14)
            ax.set_title(f"{mlabel} - {task_label}", fontsize=11)
            continue
        
        tsne = np.load(tsne_file)
        labels = np.load(label_file)
        
        benign = labels == 0
        pathogenic = labels == 1
        
        ax.scatter(tsne[benign, 0], tsne[benign, 1], 
                   c='#4A90D9', alpha=0.3, s=8, label='Benign', rasterized=True)
        ax.scatter(tsne[pathogenic, 0], tsne[pathogenic, 1], 
                   c='#E8833A', alpha=0.5, s=12, label='Pathogenic', rasterized=True)
        
        ax.set_title(f"{mlabel} — {task_label}", fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0 and j == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

plt.tight_layout()
plt.savefig(out_dir / "fig_tsne_embeddings.png", dpi=300, bbox_inches='tight')
plt.savefig(out_dir / "fig_tsne_embeddings.pdf", bbox_inches='tight')
plt.close()
print("Saved fig_tsne_embeddings.png/pdf")

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))

for i, (model, mlabel) in enumerate(zip(models, model_labels)):
    ax = axes2[i]
    tsne_file = base / f"{model}_task3_synonymous_tsne.npy"
    label_file = base / f"{model}_task3_synonymous_tsne_labels.npy"
    
    tsne = np.load(tsne_file)
    labels = np.load(label_file)
    
    benign = labels == 0
    pathogenic = labels == 1
    
    ax.scatter(tsne[benign, 0], tsne[benign, 1], 
               c='#4A90D9', alpha=0.3, s=8, label='Benign', rasterized=True)
    ax.scatter(tsne[pathogenic, 0], tsne[pathogenic, 1], 
               c='#E8833A', alpha=0.5, s=12, label='Pathogenic', rasterized=True)
    
    ax.set_title(mlabel, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

axes2[0].legend(fontsize=9, loc='upper right', framealpha=0.8)
fig2.suptitle("SynPath: t-SNE of cLM embeddings", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out_dir / "fig_tsne_synpath_only.png", dpi=300, bbox_inches='tight')
plt.savefig(out_dir / "fig_tsne_synpath_only.pdf", bbox_inches='tight')
plt.close()
print("Saved fig_tsne_synpath_only.png/pdf")