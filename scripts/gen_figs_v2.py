import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

os_path = Path('paper/figures')
os_path.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# === Data from comprehensive results ===
# Task 2 (Missense) - sorted by AUC
task2_data = [
    ("onehot_pos", "Traditional", 0.7554, 0.0107),
    ("onehot_freq", "Traditional", 0.6669, 0.0139),
    ("combined", "Traditional", 0.6662, 0.0169),
    ("CodonBERT", "cLM", 0.6602, 0.0136),
    ("CodonBERT-HF", "cLM", 0.6594, 0.0135),
    ("kmer6", "Traditional", 0.6508, 0.0245),
    ("EnCodon-80M", "cLM", 0.6329, 0.0233),
    ("kmer4", "Traditional", 0.6271, 0.0189),
    ("kmer3", "Traditional", 0.6214, 0.0203),
    ("NT-50M", "DNA LM", 0.5734, 0.0184),
    ("NT-500M", "DNA LM", 0.5695, 0.0209),
]

# Task 3 (Synonymous) - sorted by AUC
task3_data = [
    ("onehot_pos", "Traditional", 0.8910, 0.0208),
    ("CodonBERT", "cLM", 0.7336, 0.0069),
    ("CodonBERT-HF", "cLM", 0.7047, 0.0128),
    ("kmer6", "Traditional", 0.6999, 0.0290),
    ("NT-500M", "DNA LM", 0.6991, 0.0084),
    ("EnCodon-80M", "cLM", 0.6856, 0.0300),
    ("NT-50M", "DNA LM", 0.6425, 0.0192),
    ("combined", "Traditional", 0.5936, 0.0134),
    ("onehot_freq", "Traditional", 0.5890, 0.0098),
    ("kmer4", "Traditional", 0.5804, 0.0241),
    ("kmer3", "Traditional", 0.5617, 0.0212),
]

color_map = {"cLM": "#2196F3", "DNA LM": "#FF9800", "Traditional": "#9E9E9E"}

# === Fig 5: Comprehensive comparison with all baselines ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax_idx, (ax, data, title) in enumerate(zip(axes, [task2_data, task3_data], 
    ["Task 2: Missense Variant Pathogenicity", "Task 3: Synonymous Variant Pathogenicity"])):
    
    names = [d[0] for d in data]
    types = [d[1] for d in data]
    aucs = [d[2] for d in data]
    stds = [d[3] for d in data]
    colors = [color_map[t] for t in types]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, aucs, xerr=stds, color=colors, capsize=3, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('ROC-AUC')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.invert_yaxis()
    
    for i, (auc, std) in enumerate(zip(aucs, stds)):
        ax.text(auc + std + 0.01, i, f'{auc:.3f}', va='center', fontsize=9, fontweight='bold')
    
    if ax_idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[k], label=k, alpha=0.85) for k in ["cLM", "DNA LM", "Traditional"]]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('paper/figures/fig5_comprehensive_baselines.png', dpi=300, bbox_inches='tight')
print("Saved fig5")

# === Fig 6: Model type comparison (grouped) ===
fig, ax = plt.subplots(figsize=(10, 6))

categories = ["cLM\n(best)", "DNA LM\n(best)", "Traditional\n(best freq)", "Traditional\n(best pos)"]
task2_aucs = [0.6602, 0.5734, 0.6669, 0.7554]
task2_stds = [0.0136, 0.0184, 0.0139, 0.0107]
task3_aucs = [0.7336, 0.6425, 0.5890, 0.8910]
task3_stds = [0.0069, 0.0192, 0.0098, 0.0208]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, task2_aucs, width, yerr=task2_stds, label='Task 2 (Missense)', 
               color='#FF5722', capsize=4, alpha=0.85)
bars2 = ax.bar(x + width/2, task3_aucs, width, yerr=task3_stds, label='Task 3 (Synonymous)', 
               color='#3F51B5', capsize=4, alpha=0.85)

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('ROC-AUC')
ax.set_title('Best Model from Each Category: Codon LM vs DNA LM vs Traditional', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0.4, 1.0)
ax.legend(loc='upper left')

for bar, val in zip(bars1, task2_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
            ha='center', fontsize=9, fontweight='bold', color='#FF5722')
for bar, val in zip(bars2, task3_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', 
            ha='center', fontsize=9, fontweight='bold', color='#3F51B5')

plt.tight_layout()
plt.savefig('paper/figures/fig6_model_type_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig6")

# === Fig 7: Heatmap with all models ===
all_models = ["onehot_pos", "CodonBERT", "CodonBERT-HF", "EnCodon-80M", "NT-500M", "kmer6", "NT-50M", "onehot_freq", "kmer3"]
heatmap_task2 = [0.7554, 0.6602, 0.6594, 0.6329, 0.5695, 0.6508, 0.5734, 0.6669, 0.6214]
heatmap_task3 = [0.8910, 0.7336, 0.7047, 0.6856, 0.6991, 0.6999, 0.6425, 0.5890, 0.5617]

fig, ax = plt.subplots(figsize=(6, 8))
data = np.array([heatmap_task2, heatmap_task3]).T
im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=0.9, aspect='auto')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Task 2\n(Missense)', 'Task 3\n(Synonymous)'], fontsize=11)
ax.set_yticks(np.arange(len(all_models)))
ax.set_yticklabels(all_models, fontsize=10)

for i in range(len(all_models)):
    for j in range(2):
        val = data[i, j]
        color = 'white' if val < 0.55 or val > 0.78 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontweight='bold', fontsize=10)

ax.set_title('CodonBench: All Models × Tasks (Downstream, Full ClinVar)', fontsize=12, fontweight='bold')
fig.colorbar(im, ax=ax, label='ROC-AUC', shrink=0.8)
plt.tight_layout()
plt.savefig('paper/figures/fig7_full_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved fig7")

print("\nAll 3 new figures saved to paper/figures/")