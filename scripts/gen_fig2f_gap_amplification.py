import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

conditions = ['v1', 'v3a', 'v3b', 'v4', 'v2']
labels = ['v1', 'v3a', 'v3b', 'v4', 'v2']

mlp_gaps = [
    (0.719 - 0.627) * 100,
    (0.720 - 0.606) * 100,
    (0.792 - 0.647) * 100,
    (0.831 - 0.590) * 100,
    (0.643 - 0.662) * 100,
]

lr_gaps = [
    (0.701 - 0.711) * 100,
    (0.670 - 0.647) * 100,
    (0.706 - 0.675) * 100,
    (0.706 - 0.567) * 100,
    (0.669 - 0.667) * 100,
]

colors_mlp = ['#2166AC', '#2166AC', '#2166AC', '#2166AC', '#B2182B']
colors_lr = ['#92C5DE', '#92C5DE', '#92C5DE', '#92C5DE', '#EF8A62']

fig, ax = plt.subplots(figsize=(3.5, 2.8))

x = np.arange(len(conditions))
width = 0.35

bars_lr = ax.bar(x - width/2, lr_gaps, width, color=colors_lr, edgecolor='black', linewidth=0.5, label='LR gap (codon − char)')
bars_mlp = ax.bar(x + width/2, mlp_gaps, width, color=colors_mlp, edgecolor='black', linewidth=0.5, label='MLP gap (codon − char)')

for bar, val in zip(bars_mlp, mlp_gaps):
    ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}',
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=6.5, fontweight='bold')

for bar, val in zip(bars_lr, lr_gaps):
    ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
    ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}',
            ha='center', va='bottom' if val >= 0 else 'top', fontsize=6.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel('AUC gap (codon − char, pp)', fontsize=8)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.legend(fontsize=6.5, loc='upper left', framealpha=0.9)
ax.set_ylim(-6, 28)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=7)

ax.annotate('SynPath', xy=(0.98, 0.97), xycoords='axes fraction',
            ha='right', va='top', fontsize=7, fontstyle='italic', color='#333333')

fig.tight_layout()
out_dir = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'
fig.savefig(f'{out_dir}/fig2f_gap_amplification.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig2f_gap_amplification.pdf', bbox_inches='tight')
plt.close()
print('Done: fig2f_gap_amplification')