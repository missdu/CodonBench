import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\V38'
DATA = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results\unified_eval\eval_results\sklearn_seed_sweep_results.json'

d = json.load(open(DATA))
gains = {int(k.split('_')[-1]): v['gain_perfold_pp'] for k, v in d.items()}
seeds_sorted = sorted(gains.keys())
vals_sorted = [gains[s] for s in seeds_sorted]

seed42_gain = 4.7

fig, ax = plt.subplots(figsize=(6.5, 3.5))

colors = ['#D6604D' if v > 0 else '#4393C3' for v in vals_sorted]
bars = ax.bar(range(len(seeds_sorted)), vals_sorted, color=colors,
              edgecolor='black', linewidth=0.5, width=0.65)

for i, (seed, val) in enumerate(zip(seeds_sorted, vals_sorted)):
    sign = '+' if val > 0 else ''
    y = val + 0.15 if val >= 0 else val - 0.35
    ax.text(i, y, f'{sign}{val:.1f}', ha='center', fontsize=6.5,
            fontweight='bold', color='#333333')

ax.axhline(y=0, color='grey', linewidth=0.8)
mean_val = np.mean(vals_sorted)
ax.axhline(y=mean_val, color='#666666', linewidth=1.2, linestyle='--',
           label=f'Mean = {mean_val:+.1f} pp')

ax.axhline(y=seed42_gain, color='#B2182B', linewidth=1.5, linestyle=':',
           label=f'seed = 42: +{seed42_gain:.1f} pp (reported)')

ax.annotate(f'seed = 42\n+{seed42_gain:.1f} pp',
            xy=(19, seed42_gain), xytext=(15, seed42_gain + 0.8),
            fontsize=8, color='#B2182B', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#B2182B', lw=1.2))

ax.set_xticks(range(len(seeds_sorted)))
ax.set_xticklabels([str(s) for s in seeds_sorted], fontsize=7)
ax.set_xlabel('Random seed', fontsize=9)
ax.set_ylabel('LOGO-CV gain (MLP − LR, pp)', fontsize=9)
ax.set_ylim(-4, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=7, loc='upper left')

ax.text(0.98, 0.02, f'n = 20 seeds × 240 folds\n0 / 20 significant (p < 0.05)',
        transform=ax.transAxes, fontsize=7, color='#666666',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7F7F7', edgecolor='#CCCCCC'))

ax.set_title('b  sklearn MLP gain across 20 random seeds (CodonBERT, SynPath)',
             fontsize=10, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5b_seed_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig5b_seed_sweep.pdf', bbox_inches='tight')
print(f'Saved fig5b to {OUT}')