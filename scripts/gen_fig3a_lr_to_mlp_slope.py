import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

models = ['EnCodon-620M', 'CodonBERT', 'CodonTrans.', 'CodonBERT-HF', 'EnCodon-80M',
          'Mistral-117M', 'Mistral-16M', 'Mistral-1M', 'CaLM', 'cdsBERT',
          'ESM-2', 'ESM-1b']
tok_type = ['codon']*9 + ['char'] + ['pLM']*2

syn_lr =  [0.785, 0.7336, 0.713, 0.7047, 0.6856, 0.6564, 0.5984, 0.5914, 0.6725, 0.5979, 0.6797, 0.6818]
syn_mlp = [0.800, 0.8486, 0.8178, 0.816, 0.8115, 0.7998, 0.841, 0.734, 0.7825, 0.5819, 0.6178, 0.6651]
mis_lr =  [0.6167, 0.6602, 0.6944, 0.6594, 0.6329, 0.6112, 0.671, 0.6656, 0.6886, 0.6288, 0.7191, 0.7102]
mis_mlp = [0.6667, 0.6772, 0.6796, 0.6499, 0.6741, 0.5741, 0.6761, 0.6708, 0.6922, 0.609, 0.6919, 0.7072]

color_map = {'codon': '#2166AC', 'char': '#E8833A', 'pLM': '#4DAF4A'}
colors = [color_map[t] for t in tok_type]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

for i, model in enumerate(models):
    ax1.plot([0, 1], [syn_lr[i], syn_mlp[i]], '-o', color=colors[i],
             linewidth=1.5, markersize=5, alpha=0.85, zorder=3)
    ax2.plot([0, 1], [mis_lr[i], mis_mlp[i]], '-o', color=colors[i],
             linewidth=1.5, markersize=5, alpha=0.85, zorder=3)

for ax, title in [(ax1, 'SynPath'), (ax2, 'MisPath')]:
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['LR', 'MLP'], fontsize=9)
    ax.set_ylabel('AUC', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(0.50, 0.92)
    ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='#2166AC', linewidth=2, marker='o', markersize=4, label='Codon-tokenized (n=9)'),
                   Line2D([0], [0], color='#E8833A', linewidth=2, marker='o', markersize=4, label='Character-tokenized (n=1)'),
                   Line2D([0], [0], color='#4DAF4A', linewidth=2, marker='o', markersize=4, label='Amino-acid-tokenized (n=2)')]
ax2.legend(handles=legend_elements, fontsize=6.5, loc='lower center', framealpha=0.9)

ax1.annotate('codon cLMs rise\n+1.5 to +24.3 pp', xy=(0.45, 0.87), fontsize=6.5,
             ha='center', color='#2166AC', fontstyle='italic')
ax1.annotate('pLMs fall\n-6.2 to -1.7 pp', xy=(0.45, 0.60), fontsize=6.5,
             ha='center', color='#4DAF4A', fontstyle='italic')
ax1.annotate('cdsBERT falls\n-1.6 pp', xy=(0.45, 0.52), fontsize=6.5,
             ha='center', color='#E8833A', fontstyle='italic')
ax2.annotate('all models\n~ flat', xy=(0.45, 0.72), fontsize=6.5,
             ha='center', color='#333333', fontstyle='italic')

fig.tight_layout(w_pad=3)
out_dir = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'
fig.savefig(f'{out_dir}/fig3a_lr_to_mlp_slope.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig3a_lr_to_mlp_slope.pdf', bbox_inches='tight')
plt.close()
print('Done: fig3a_lr_to_mlp_slope (12 models, 3 colors)')
