import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

tiers = ['LLR\n(Level 0)', 'LR\n(Level 1)', 'MLP\n(Level 2)', 'LoRA\n(Level 3)']
x = np.arange(len(tiers))

CODON_C = '#1565C0'
CHAR_C = '#E65100'
SYNTH_C = '#757575'

codon_shades = {'v1': '#42A5F5', 'v3a': '#1E88E5', 'v3b': '#1565C0', 'v4': '#0D47A1', 'v2': '#757575'}
char_shades = {'v1': '#FFA726', 'v3a': '#FB8C00', 'v3b': '#E65100', 'v4': '#BF360C', 'v2': '#757575'}

markers = {'v1': 'o', 'v3a': 's', 'v3b': 'D', 'v4': '^', 'v2': 'P'}
linestyles = {'v1': ':', 'v3a': '--', 'v3b': '-', 'v4': '-.', 'v2': '--'}

versions = [
    ('v1',  [None, 0.701, 0.719, None], [None, 0.711, 0.627, None]),
    ('v3a', [None, 0.670, 0.720, None], [None, 0.647, 0.606, None]),
    ('v3b', [None, 0.706, 0.792, None], [None, 0.675, 0.647, None]),
    ('v4',  [None, 0.706, 0.831, None], [None, 0.567, 0.590, None]),
    ('v2',  [None, 0.669, 0.643, None], [None, 0.667, 0.662, None]),
]

for label, codon_vals, char_vals in versions:
    m = markers[label]
    ls = linestyles[label]

    cx = [x[i] for i in range(4) if codon_vals[i] is not None]
    cy = [codon_vals[i] for i in range(4) if codon_vals[i] is not None]
    if cx:
        ax.plot(cx, cy, marker=m, color=codon_shades[label], linewidth=2.2, markersize=8,
                linestyle=ls, zorder=3)

    chx = [x[i] for i in range(4) if char_vals[i] is not None]
    chy = [char_vals[i] for i in range(4) if char_vals[i] is not None]
    if chx:
        ax.plot(chx, chy, marker=m, color=char_shades[label], linewidth=2.2, markersize=8,
                linestyle=ls, zorder=3)

legend_elements = [
    Line2D([0], [0], color=CODON_C, linewidth=2.5, marker='None', linestyle='-', label='Codon tokenizer'),
    Line2D([0], [0], color=CHAR_C, linewidth=2.5, marker='None', linestyle='-', label='Character tokenizer'),
    Line2D([0], [0], color='black', linewidth=0, marker='o', markersize=6, label='v1 (4.8K CDS, 10ep)'),
    Line2D([0], [0], color='black', linewidth=0, marker='s', markersize=6, label='v3a (4.8K CDS, 100ep)'),
    Line2D([0], [0], color='black', linewidth=0, marker='D', markersize=6, label='v3b (114K CDS, 30ep)'),
    Line2D([0], [0], color='black', linewidth=0, marker='^', markersize=6, label='v4 (114K CDS, 110M)'),
    Line2D([0], [0], color=SYNTH_C, linewidth=0, marker='P', markersize=6, label='v2 (synth., neg. ctrl.)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=7.5, framealpha=0.9, ncol=1)

ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=10)
ax.set_ylabel('SynPath AUC', fontsize=11)
ax.set_title('Codon vs Character: the gap appears only at nonlinear depth', fontsize=12, fontweight='bold')

ax.set_ylim(0.52, 0.88)
ax.set_xlim(-0.3, 3.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26\fig_extractable_depth_codon_char.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig_extractable_depth_codon_char')
