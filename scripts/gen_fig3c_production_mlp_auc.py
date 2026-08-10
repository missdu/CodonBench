import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

models = ['EnCodon\n620M', 'Codon\nBERT', 'Codon\nTrans.', 'CodonBERT\n-HF', 'EnCodon\n80M',
          'Mistral\n117M', 'Mistral\n16M', 'Mistral\n1M', 'CaLM', 'cdsBERT',
          'ESM-2', 'ESM-1b']
tok_type = ['codon']*9 + ['char'] + ['pLM']*2

syn_lr  = [0.785, 0.7336, 0.713, 0.7047, 0.6856, 0.6564, 0.5984, 0.5914, 0.6725, 0.5979, 0.6797, 0.6818]
syn_mlp = [0.800, 0.8486, 0.8178, 0.816, 0.8115, 0.7998, 0.841, 0.734, 0.7825, 0.5819, 0.6178, 0.6651]
mis_lr  = [0.6167, 0.6602, 0.6944, 0.6594, 0.6329, 0.6112, 0.671, 0.6656, 0.6886, 0.6288, 0.7191, 0.7102]
mis_mlp = [0.6667, 0.6772, 0.6796, 0.6499, 0.6741, 0.5741, 0.6761, 0.6708, 0.6922, 0.609, 0.6919, 0.7072]

x = np.arange(len(models))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.2))

lr_color_map = {'codon': '#92C5DE', 'char': '#FDB863', 'pLM': '#A6D96A'}
mlp_color_map = {'codon': '#2166AC', 'char': '#E8833A', 'pLM': '#4DAF4A'}
colors_lr = [lr_color_map[t] for t in tok_type]
colors_mlp = [mlp_color_map[t] for t in tok_type]

for ax, lr_vals, mlp_vals, title in [(ax1, syn_lr, syn_mlp, 'SynPath'),
                                       (ax2, mis_lr, mis_mlp, 'MisPath')]:
    ax.bar(x - width/2, lr_vals, width, color=colors_lr, edgecolor='black', linewidth=0.5, label='LR')
    ax.bar(x + width/2, mlp_vals, width, color=colors_mlp, edgecolor='black', linewidth=0.5, label='MLP')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=5)
    ax.set_ylabel('AUC', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0.45, 0.92)
    ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)

from matplotlib.patches import Patch
legend_codon = Patch(facecolor='#2166AC', edgecolor='black', label='Codon-tokenized')
legend_char = Patch(facecolor='#E8833A', edgecolor='black', label='Character-tokenized')
legend_plm = Patch(facecolor='#4DAF4A', edgecolor='black', label='Amino-acid-tokenized')
ax1.legend(handles=[legend_codon, legend_char, legend_plm], fontsize=6, loc='upper right', framealpha=0.9)

fig.tight_layout(w_pad=3)
out_dir = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'
fig.savefig(f'{out_dir}/fig3c_production_mlp_auc.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig3c_production_mlp_auc.pdf', bbox_inches='tight')
plt.close()
print('Done: fig3c_production_mlp_auc (10 models, 2 colors)')
