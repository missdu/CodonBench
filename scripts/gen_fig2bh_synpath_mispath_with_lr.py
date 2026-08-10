import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

CC, CH = "#2166AC", "#E08214"
w = 0.18

versions = ['v1', 'v3a', 'v3b', 'v4', 'v2']
x = np.arange(len(versions))

# === Left: SynPath ===
lr_codon = [0.701, 0.670, 0.706, 0.706, 0.669]
lr_char  = [0.711, 0.647, 0.675, 0.567, 0.667]
mlp_codon = [0.719, 0.720, 0.792, 0.831, 0.643]
mlp_char  = [0.627, 0.606, 0.647, 0.590, 0.662]
lr_codon_std = [0.013, 0.018, 0.013, 0.012, 0.011]
lr_char_std  = [0.023, 0.026, 0.004, 0.017, 0.016]

for i in range(len(versions)):
    if lr_codon[i] is not None:
        ax1.bar(x[i] - 1.5*w, lr_codon[i], w, color='#90CAF9', edgecolor='gray', linewidth=0.5,
                label='Codon LR' if i == 0 else '')
        if lr_codon_std[i] is not None:
            ax1.errorbar(x[i] - 1.5*w, lr_codon[i], yerr=lr_codon_std[i], fmt='none', ecolor='gray', capsize=2, linewidth=0.8)
    if lr_char[i] is not None:
        ax1.bar(x[i] - 0.5*w, lr_char[i], w, color='#FFCC80', edgecolor='gray', linewidth=0.5,
                label='Char LR' if i == 0 else '')
        if lr_char_std[i] is not None:
            ax1.errorbar(x[i] - 0.5*w, lr_char[i], yerr=lr_char_std[i], fmt='none', ecolor='gray', capsize=2, linewidth=0.8)
    if mlp_codon[i] is not None:
        ax1.bar(x[i] + 0.5*w, mlp_codon[i], w, color=CC, edgecolor='gray', linewidth=0.5,
                label='Codon MLP' if i == 0 else '')
    if mlp_char[i] is not None:
        ax1.bar(x[i] + 1.5*w, mlp_char[i], w, color=CH, edgecolor='gray', linewidth=0.5,
                label='Char MLP' if i == 0 else '')

deltas = {0: 9.2, 1: 11.4, 2: 14.5, 3: 24.1}
for i, delta in deltas.items():
    if mlp_codon[i] is not None and mlp_char[i] is not None:
        ax1.annotate('', xy=(x[i]+0.5*w, mlp_codon[i]-0.008), xytext=(x[i]+1.5*w, mlp_char[i]-0.008),
                    arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5))
        ax1.text(x[i]+w, 0.75, f'+{delta:.1f}pp',
                ha='center', fontsize=8, fontweight='bold', color='#C62828')

ax1.text(x[3]+0.5*w, 0.831+0.012, f'{0.831:.3f}', ha='center', fontsize=7.5, fontweight='bold', color=CC)
ax1.text(x[4], 0.615, 'No gap\n(synthetic)', ha='center', fontsize=7, color='#888', style='italic')

ax1.axvline(x=3.5, color='#CCCCCC', linewidth=1, linestyle='--')
ax1.set_xticks(x)
ax1.set_xticklabels(versions, fontsize=10)
ax1.set_ylabel('AUC', fontsize=11)
ax1.set_title('SynPath (synonymous)', fontsize=12, fontweight='bold')
ax1.set_ylim(0.58, 0.88)
ax1.legend(fontsize=8, ncol=2, loc='upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# === Right: MisPath (LR + MLP) ===
lr_codon_mis = [0.704, 0.701, 0.691, 0.680, 0.683]
lr_char_mis  = [0.701, 0.675, 0.719, 0.653, 0.684]
mlp_codon_mis = [0.655, 0.670, 0.674, 0.681, 0.648]
mlp_char_mis  = [0.648, 0.634, 0.702, 0.643, 0.661]

for i in range(len(versions)):
    if lr_codon_mis[i] is not None:
        ax2.bar(x[i] - 1.5*w, lr_codon_mis[i], w, color='#90CAF9', edgecolor='gray', linewidth=0.5,
                label='Codon LR' if i == 0 else '')
    if lr_char_mis[i] is not None:
        ax2.bar(x[i] - 0.5*w, lr_char_mis[i], w, color='#FFCC80', edgecolor='gray', linewidth=0.5,
                label='Char LR' if i == 0 else '')
    if mlp_codon_mis[i] is not None:
        ax2.bar(x[i] + 0.5*w, mlp_codon_mis[i], w, color=CC, edgecolor='gray', linewidth=0.5,
                label='Codon MLP' if i == 0 else '')
    if mlp_char_mis[i] is not None:
        ax2.bar(x[i] + 1.5*w, mlp_char_mis[i], w, color=CH, edgecolor='gray', linewidth=0.5,
                label='Char MLP' if i == 0 else '')

ax2.annotate('', xy=(x[2]+1.5*w, 0.702-0.008), xytext=(x[2]+0.5*w, 0.674-0.008),
            arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5))
ax2.text(x[2]+w, 0.75, 'char > codon\n(reversed)', ha='center', fontsize=8, fontweight='bold', color='#C62828')

ax2.text(x[4], 0.615, 'No gap\n(synthetic)', ha='center', fontsize=7, color='#888', style='italic')

ax2.axvline(x=3.5, color='#CCCCCC', linewidth=1, linestyle='--')
ax2.set_xticks(x)
ax2.set_xticklabels(versions, fontsize=10)
ax2.set_ylabel('AUC', fontsize=11)
ax2.set_title('MisPath (missense)', fontsize=12, fontweight='bold')
ax2.set_ylim(0.58, 0.88)
ax2.legend(fontsize=8, ncol=2, loc='upper left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
out_path = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26\fig2bh_synpath_mispath_with_lr.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print('Saved fig2bh_synpath_mispath_with_lr')