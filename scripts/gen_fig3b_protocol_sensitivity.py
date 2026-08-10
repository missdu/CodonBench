import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

models = ['CaLM', 'cdsBERT', 'CodonBERT', 'CodonBERT-HF', 'CodonTrans.',
          'EnCodon-620M', 'EnCodon-80M', 'Mistral-117M', 'Mistral-16M', 'Mistral-1M']

syn_lr  = [0.6725, 0.5979, 0.7336, 0.7047, 0.713, 0.785, 0.6856, 0.6564, 0.5984, 0.5914]
syn_mlp = [0.7825, 0.5819, 0.8486, 0.816, 0.8178, 0.8, 0.8115, 0.7998, 0.841, 0.734]
mis_lr  = [0.6886, 0.6288, 0.6602, 0.6594, 0.6944, 0.6167, 0.6329, 0.6112, 0.671, 0.6656]
mis_mlp = [0.6922, 0.609, 0.6772, 0.6499, 0.6796, 0.6667, 0.6741, 0.5741, 0.6761, 0.6708]

fig, ax = plt.subplots(figsize=(4.5, 4))

ax.scatter(mis_lr, mis_mlp, c='#2166AC', s=50, alpha=0.8, edgecolors='black', linewidths=0.5,
           zorder=3, label='MisPath (ρ = 0.855, p = 0.002)')
ax.scatter(syn_lr, syn_mlp, c='#E8833A', s=50, alpha=0.8, edgecolors='black', linewidths=0.5,
           marker='D', zorder=3, label='SynPath (ρ = 0.588, p = 0.074)')

lim_min, lim_max = 0.55, 0.90
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.8, alpha=0.4, zorder=1)

ax.set_xlabel('LR AUC', fontsize=9)
ax.set_ylabel('MLP AUC', fontsize=9)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)

ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

ax.annotate('Δρ = 0.267', xy=(0.97, 0.03), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='grey', alpha=0.9))

fig.tight_layout()
out_dir = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'
fig.savefig(f'{out_dir}/fig3b_protocol_sensitivity.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out_dir}/fig3b_protocol_sensitivity.pdf', bbox_inches='tight')
plt.close()
print('Done: fig3b_protocol_sensitivity (10 models)')
