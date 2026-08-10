import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6),
                                      gridspec_kw={'width_ratios': [1.2, 1.0, 1.2]})
plt.subplots_adjust(wspace=0.35)

models = ['CB-HF', 'CB', 'EC-80M']
model_colors = ['#08519C', '#4292C6', '#9ECAE1']

# ============================================================
# Panel A: Ridge vs MLP paired bar chart (4 regression tasks)
# ============================================================
tasks = ['mRFP', 'E.coli', 'Fungal', 'mRNA\nstability']
task_keys = ['mrfp', 'ecoli', 'fungal', 'mrna']

ridge_data = {
    'mrfp':  [0.253, -0.022, 0.124],
    'ecoli': [-0.397, -0.308, -0.455],
    'fungal':[0.563, 0.544, 0.483],
    'mrna':  [0.011, -0.013, -0.151],
}
mlp_data = {
    'mrfp':  [0.463, 0.041, 0.361],
    'ecoli': [-0.161, -0.106, -0.008],
    'fungal':[0.631, 0.613, 0.643],
    'mrna':  [0.019, 0.003, 0.031],
}

n_tasks = len(tasks)
n_models = len(models)
group_w = 0.7
bar_w = group_w / (2 * n_models)

for ti, (task, key) in enumerate(zip(tasks, task_keys)):
    for mi in range(n_models):
        x_ridge = ti - group_w/2 + mi * 2 * bar_w + bar_w/2
        x_mlp = x_ridge + bar_w

        r_val = ridge_data[key][mi]
        m_val = mlp_data[key][mi]

        ax1.bar(x_ridge, r_val, bar_w * 0.9, color='#92C5DE', edgecolor='black', linewidth=0.3)
        ax1.bar(x_mlp, m_val, bar_w * 0.9, color='#2166AC', edgecolor='black', linewidth=0.3)

        delta = (m_val - r_val) * 100
        if abs(delta) > 5:
            y_pos = max(r_val, m_val) + 0.02
            color = '#B2182B' if delta > 0 else '#2166AC'
            ax1.text((x_ridge + x_mlp) / 2, y_pos, f'+{delta:.0f}',
                     ha='center', fontsize=5.5, fontweight='bold', color=color)

ax1.set_xticks(range(n_tasks))
ax1.set_xticklabels(tasks, fontsize=9)
ax1.set_ylabel('R²', fontsize=10)
ax1.axhline(y=0, color='grey', linewidth=0.8)
ax1.set_ylim(-0.55, 0.75)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#92C5DE', edgecolor='black', linewidth=0.5, label='Ridge'),
                   Patch(facecolor='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')]
ax1.legend(handles=legend_elements, fontsize=8, loc='upper left')
ax1.set_title('a  Ridge vs MLP probing', fontsize=11, fontweight='bold', loc='left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.annotate('codon-determined', xy=(0.25, 0.97), xycoords='axes fraction',
             ha='center', va='top', fontsize=7, color='#2166AC', fontstyle='italic')
ax1.annotate('protein-determined', xy=(0.75, 0.97), xycoords='axes fraction',
             ha='center', va='top', fontsize=7, color='#B2182B', fontstyle='italic')

# ============================================================
# Panel B: Four-level probing protocol staircase
# ============================================================
protocols = ['LR', 'MLP', 'LoRA', 'Fine-\ntune']
prot_x = np.arange(len(protocols))

synpath_vals = [0.668, 0.816, 0.880, 0.912]
mispath_vals = [0.646, 0.650, 0.814, 0.840]
mrfp_vals    = [0.253, 0.463, None, None]
fungal_vals  = [0.563, 0.631, None, None]

ax2.plot(prot_x[:2], synpath_vals[:2], 'o-', color='#2166AC', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='SynPath (AUC)')
ax2.plot(prot_x, mispath_vals, 's--', color='#E08214', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='MisPath (AUC)')
ax2.plot(prot_x[:2], [v/1.0 for v in mrfp_vals[:2]], '^-', color='#4DAF4A', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='mRFP (R², ×1)')
ax2.plot(prot_x[:2], [v/1.0 for v in fungal_vals[:2]], 'D-', color='#984EA3', linewidth=2, markersize=8,
         markerfacecolor='white', markeredgewidth=2, label='Fungal (R², ×1)')

for i, v in enumerate(synpath_vals[:2]):
    ax2.text(prot_x[i], v + 0.02, f'{v:.3f}', ha='center', fontsize=7, color='#2166AC', fontweight='bold')
for i, v in enumerate(mispath_vals):
    ax2.text(prot_x[i], v - 0.03, f'{v:.3f}', ha='center', fontsize=7, color='#E08214', fontweight='bold')

ax2.set_xticks(prot_x)
ax2.set_xticklabels(protocols, fontsize=9)
ax2.set_ylabel('Performance', fontsize=10)
ax2.set_ylim(0.1, 1.0)
ax2.legend(fontsize=7, loc='upper left', framealpha=0.9)
ax2.set_title('b  Probing depth staircase', fontsize=11, fontweight='bold', loc='left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.annotate('Deeper probing →\ncodon-determined tasks\nimprove more',
             xy=(1.5, 0.25), fontsize=7, fontstyle='italic', color='#333',
             ha='center')

# ============================================================
# Panel C: Heatmap matrix — Task × Protocol
# ============================================================
task_labels = ['SynPath', 'MisPath', 'mRFP', 'E.coli', 'Fungal', 'mRNA\nstab.']
prot_labels = ['LR', 'MLP', 'LoRA']

heatmap_data = np.array([
    [0.668, 0.816, 0.880],
    [0.646, 0.650, 0.814],
    [0.253, 0.463, np.nan],
    [-0.397, -0.161, np.nan],
    [0.563, 0.631, np.nan],
    [0.011, 0.019, np.nan],
])

lr_to_mlp_gain = np.array([
    (0.816 - 0.668) * 100,
    (0.650 - 0.646) * 100,
    (0.463 - 0.253) * 100,
    (-0.161 - (-0.397)) * 100,
    (0.631 - 0.563) * 100,
    (0.019 - 0.011) * 100,
])

im = ax3.imshow(lr_to_mlp_gain.reshape(-1, 1), cmap='RdYlBu_r', aspect='auto',
                vmin=-5, vmax=25)

ax3.set_xticks([0])
ax3.set_xticklabels(['LR→MLP\ngain (pp)'], fontsize=9)
ax3.set_yticks(range(len(task_labels)))
ax3.set_yticklabels(task_labels, fontsize=9)

for i in range(len(task_labels)):
    val = lr_to_mlp_gain[i]
    color = 'white' if val > 15 else 'black'
    ax3.text(0, i, f'{val:+.1f}', ha='center', va='center', fontsize=10, fontweight='bold', color=color)

cbar = fig.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
cbar.set_label('LR→MLP gain (pp)', fontsize=9)

ax3.set_title('c  Probing-depth dependency', fontsize=11, fontweight='bold', loc='left')

ax3.annotate('codon-determined', xy=(1.15, 0.25), xycoords='axes fraction',
             ha='center', va='center', fontsize=7, color='#2166AC', fontstyle='italic', rotation=90)
ax3.annotate('protein-determined', xy=(1.15, 0.75), xycoords='axes fraction',
             ha='center', va='center', fontsize=7, color='#B2182B', fontstyle='italic', rotation=90)

fig.savefig(f'{OUT}/fig_regression_overview.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig_regression_overview.pdf', bbox_inches='tight')
plt.close()
print('Done: fig_regression_overview (3 panels)')