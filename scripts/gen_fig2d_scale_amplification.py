import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

fig, ax1 = plt.subplots(1, 1, figsize=(8, 5.5))

BLUE_C = '#2166AC'
ORANGE_C = '#E08214'

# === Blue x-axis: data scale (training samples) ===
samples = [48420, 484200, 3423570]
deltas = [9.2, 11.4, 14.5]
ci_lower = [3.5, 6.1, 9.7]
ci_upper = [14.9, 16.7, 19.3]

for i in range(3):
    err = [[deltas[i] - ci_lower[i]], [ci_upper[i] - deltas[i]]]
    ax1.errorbar(samples[i], deltas[i], yerr=err, fmt='o', color=BLUE_C,
                markersize=10, capsize=5, linewidth=2, markerfacecolor='white', markeredgewidth=2, zorder=3)

ax1.plot(samples, deltas, '--', color=BLUE_C, linewidth=1.5, alpha=0.5, zorder=1)

ax1.scatter([1650000], [-1.9], marker='X', color='#BDBDBD', s=80, zorder=3)

ax1.set_xscale('log')
ax1.set_xlabel('Training samples (data-scale layer)', fontsize=11, color=BLUE_C, fontweight='bold')
ax1.set_ylabel('Codon-Character Delta MLP AUC (pp)', fontsize=11)
ax1.tick_params(axis='x', labelcolor=BLUE_C)

ax1.text(samples[0], deltas[0]+1.5, 'v1', ha='center', fontsize=9, color=BLUE_C, fontweight='bold')
ax1.text(samples[1], deltas[1]+1.5, 'v3a', ha='center', fontsize=9, color=BLUE_C, fontweight='bold')
ax1.text(samples[2], deltas[2]+1.5, 'v3b', ha='center', fontsize=9, color=BLUE_C, fontweight='bold')
ax1.text(1650000, -1.9-1.5, 'v2 (synth.)', ha='center', fontsize=8, color='#999', style='italic')

# === Orange x-axis: model scale (params) ===
ax2 = ax1.twiny()

params = [20, 110]
ax2.scatter([20], [14.5], marker='D', color=ORANGE_C, s=100, zorder=4, facecolors='white', linewidths=2)
ax2.annotate('', xy=(110, 24.1), xytext=(20, 14.5),
            arrowprops=dict(arrowstyle='->', color=ORANGE_C, lw=2.5))

v4_delta = 24.1
v4_ci_lower = 18.7
v4_ci_upper = 29.5
v4_err = [[v4_delta - v4_ci_lower], [v4_ci_upper - v4_delta]]
ax2.errorbar([110], [v4_delta], yerr=v4_err, fmt='^', color=ORANGE_C,
            markersize=10, capsize=5, linewidth=2, markerfacecolor='white', markeredgewidth=2, zorder=4)
ax2.text(110, v4_ci_upper + 1.2, 'v4: codon MLP 0.792→0.831 (+3.9pp)\nchar MLP 0.647→0.590 (−5.7pp)\nGap: 14.5→24.1pp (95% CI [18.7, 29.5])', ha='center', fontsize=8,
        color=ORANGE_C, fontweight='bold')
ax2.text(20, 14.5-2, 'v3b', ha='center', fontsize=9, color=ORANGE_C, fontweight='bold')

ax2.set_xscale('log')
ax2.set_xlabel('Model parameters (model-scale layer)', fontsize=11, color=ORANGE_C, fontweight='bold')
ax2.tick_params(axis='x', labelcolor=ORANGE_C)
ax2.set_xlim(15, 150)

ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1)
ax1.set_ylim(-6, 36)
ax1.set_title('Scale amplification: data scale and model scale', fontsize=12, fontweight='bold', pad=25)
ax1.spines['right'].set_visible(False)

legend_elements = [
    Line2D([0], [0], color=BLUE_C, linewidth=2, marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2, label='Data-scale layer (v1, v3a, v3b)'),
    Line2D([0], [0], color=ORANGE_C, linewidth=2.5, marker='^', markersize=8, markerfacecolor='white', markeredgewidth=2, label='Model-scale layer (v3b -> v4)'),
    Line2D([0], [0], color='#BDBDBD', linewidth=0, marker='X', markersize=8, label='v2 (synth., neg. ctrl.)'),
]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper left', framealpha=0.9)

plt.tight_layout()
out_path = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26\fig2d_scale_amplification.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig2d_scale_amplification')
