import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

CC, CH = "#2166AC", "#E08214"

versions = ['v1', 'v3a', 'v3b', 'v4', 'v2']
x = np.arange(len(versions))
w = 0.18

# Raw data (None = missing)
lr_codon_raw = [0.701, 0.670, 0.706, None, 0.669]
lr_char_raw  = [0.711, 0.647, 0.675, None, 0.667]
mlp_codon_raw = [0.719, 0.720, 0.792, 0.831, 0.643]
mlp_char_raw  = [0.627, 0.606, 0.647, None, 0.662]
lr_codon_std_raw = [0.013, 0.018, 0.013, None, 0.011]
lr_char_std_raw  = [0.023, 0.026, 0.004, None, 0.016]

# Replace None with 0 for plotting, track which to hide
def safe(vals):
    return [v if v is not None else 0 for v in vals]

def mask(vals):
    return [v is not None for v in vals]

lr_codon = safe(lr_codon_raw)
lr_char = safe(lr_char_raw)
mlp_codon = safe(mlp_codon_raw)
mlp_char = safe(mlp_char_raw)

# LR bars
for i in range(len(versions)):
    if lr_codon_raw[i] is not None:
        ax.bar(x[i] - 1.5*w, lr_codon_raw[i], w, color='#90CAF9', edgecolor='gray', linewidth=0.5,
               label='Codon LR' if i == 0 else '')
        if lr_codon_std_raw[i] is not None:
            ax.errorbar(x[i] - 1.5*w, lr_codon_raw[i], yerr=lr_codon_std_raw[i], fmt='none', ecolor='gray', capsize=2, linewidth=0.8)
    if lr_char_raw[i] is not None:
        ax.bar(x[i] - 0.5*w, lr_char_raw[i], w, color='#FFCC80', edgecolor='gray', linewidth=0.5,
               label='Char LR' if i == 0 else '')
        if lr_char_std_raw[i] is not None:
            ax.errorbar(x[i] - 0.5*w, lr_char_raw[i], yerr=lr_char_std_raw[i], fmt='none', ecolor='gray', capsize=2, linewidth=0.8)

# MLP bars
for i in range(len(versions)):
    if mlp_codon_raw[i] is not None:
        ax.bar(x[i] + 0.5*w, mlp_codon_raw[i], w, color=CC, edgecolor='gray', linewidth=0.5,
               label='Codon MLP' if i == 0 else '')
    if mlp_char_raw[i] is not None:
        ax.bar(x[i] + 1.5*w, mlp_char_raw[i], w, color=CH, edgecolor='gray', linewidth=0.5,
               label='Char MLP' if i == 0 else '')

# Red arrows for delta MLP
deltas = {0: 9.2, 1: 11.4, 2: 14.5}
for i, delta in deltas.items():
    if mlp_codon_raw[i] is not None and mlp_char_raw[i] is not None:
        ax.annotate('', xy=(x[i]+0.5*w, mlp_codon_raw[i]-0.008), xytext=(x[i]+1.5*w, mlp_char_raw[i]-0.008),
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5))
        ax.text(x[i]+w, min(mlp_codon_raw[i], mlp_char_raw[i])-0.022, f'+{delta:.1f}pp',
                ha='center', fontsize=8, fontweight='bold', color='#C62828')

# v4: only codon MLP available
ax.text(x[3]+0.5*w, mlp_codon_raw[3]+0.012, f'{mlp_codon_raw[3]:.3f}', ha='center', fontsize=7.5, fontweight='bold', color=CC)
ax.text(x[3]+1.5*w, 0.65, 'char\npending', ha='center', fontsize=7, color='#999', style='italic')

# v2 annotation
ax.text(x[4], 0.615, 'No gap\n(synthetic)', ha='center', fontsize=7, color='#888', style='italic')

# Divider
ax.axvline(x=3.5, color='#CCCCCC', linewidth=1, linestyle='--')
ax.text(3.5, 0.855, 'neg. ctrl.', ha='center', fontsize=7, color='#999', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=10)
ax.set_ylabel('SynPath AUC', fontsize=11)
ax.set_ylim(0.58, 0.88)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26\fig2b_lr_vs_mlp.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig2b_lr_vs_mlp')
