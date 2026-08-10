"""Generate V15 new figures:
1. From-scratch ablation: tokenization × probing-depth interaction
2. AlphaMissense channel separation confirmation
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})

OUT_DIR = 'paper/figures_v2/'

# ============================================================
# Fig A: From-scratch ablation — tokenization × probing-depth
# ============================================================

versions = ['v1\n(4.8K real\n10ep)', 'v3a\n(4.8K real\n100ep)', 'v3b\n(114K real\n30ep)']

codon_lr = [0.701, 0.670, 0.706]
char_lr = [0.711, 0.647, 0.675]

codon_mlp = [0.719, 0.720, 0.792]
char_mlp = [0.627, 0.606, 0.647]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

x = np.arange(len(versions))
width = 0.35

# Left panel: LR
bars1 = axes[0].bar(x - width/2, codon_lr, width, label='Codon tokenizer', color='#2166ac', edgecolor='white', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, char_lr, width, label='Character tokenizer', color='#b2182b', edgecolor='white', linewidth=0.5)
axes[0].set_ylabel('AUC')
axes[0].set_title('Linear Probing (LR)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(versions)
axes[0].legend(loc='lower right')
axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
axes[0].set_ylim(0.55, 0.85)

for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# Add delta annotations
for i in range(len(versions)):
    delta = codon_lr[i] - char_lr[i]
    sign = '+' if delta >= 0 else ''
    axes[0].annotate(f'Δ={sign}{delta:.1f}pp',
                    xy=(x[i], max(codon_lr[i], char_lr[i]) + 0.02),
                    ha='center', fontsize=8, color='#666666',
                    fontstyle='italic')

# Right panel: MLP
bars3 = axes[1].bar(x - width/2, codon_mlp, width, label='Codon tokenizer', color='#2166ac', edgecolor='white', linewidth=0.5)
bars4 = axes[1].bar(x + width/2, char_mlp, width, label='Character tokenizer', color='#b2182b', edgecolor='white', linewidth=0.5)
axes[1].set_ylabel('AUC')
axes[1].set_title('Nonlinear Probing (MLP)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(versions)
axes[1].legend(loc='lower right')
axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
axes[1].set_ylim(0.55, 0.85)

for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars4:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# Add delta annotations with emphasis
for i in range(len(versions)):
    delta = (codon_mlp[i] - char_mlp[i]) * 100
    axes[1].annotate(f'Δ=+{delta:.1f}pp',
                    xy=(x[i], max(codon_mlp[i], char_mlp[i]) + 0.02),
                    ha='center', fontsize=9, fontweight='bold', color='#2166ac')

# Add scale amplification arrow
axes[1].annotate('', xy=(2.3, 0.79), xytext=(0.3, 0.73),
                arrowprops=dict(arrowstyle='->', color='#4393c3', lw=2))
axes[1].text(1.3, 0.77, 'Scale\namplification', ha='center', fontsize=8,
            color='#4393c3', fontstyle='italic')

fig.suptitle('From-Scratch Ablation: Tokenization Determines Extractable Depth\n(Same BERT architecture, same corpus, only tokenizer differs)',
            fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(OUT_DIR + 'fig_ablation_depth.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR + 'fig_ablation_depth.pdf', bbox_inches='tight')
plt.close()
print(f"Saved {OUT_DIR}fig_ablation_depth.png/pdf")


# ============================================================
# Fig B: AlphaMissense channel separation
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

models = ['AlphaMissense', 'ESM-2-650M', 'ESM-1b', 'CodonBERT', 'CodonBERT-HF', 'EnCodon-80M', 'onehot_pos†']
missense_auc = [0.956, 0.719, 0.711, 0.660, 0.659, 0.633, 0.755]
synonymous_auc = [0.440, 0.680, 0.600, 0.734, 0.705, 0.686, 0.891]

x = np.arange(len(models))
width = 0.35

bars_miss = ax.bar(x - width/2, missense_auc, width, label='Task 1: Missense (I(A;Y) dominant)',
                   color='#b2182b', edgecolor='white', linewidth=0.5, alpha=0.85)
bars_syn = ax.bar(x + width/2, synonymous_auc, width, label='Task 2: Synonymous (I(σ;Y|A) pure)',
                  color='#2166ac', edgecolor='white', linewidth=0.5, alpha=0.85)

ax.set_ylabel('AUC (5-fold CV)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')
ax.legend(loc='upper right')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_ylim(0.3, 1.05)

# Annotate AlphaMissense
ax.annotate('Below chance!\nNo access to I(σ;Y|A)',
           xy=(0, 0.440), xytext=(0.5, 0.35),
           arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5),
           fontsize=9, color='#b2182b', fontweight='bold', ha='center')

# Annotate AlphaMissense missense
ax.annotate('0.956',
           xy=(0 - width/2, 0.956), xytext=(0 - width/2, 0.98),
           fontsize=9, fontweight='bold', color='#b2182b', ha='center')

ax.set_title('AlphaMissense Confirms Channel Separation:\nNear-Perfect on Missense, Below Chance on Synonymous',
            fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR + 'fig_alphamissense_channels.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR + 'fig_alphamissense_channels.pdf', bbox_inches='tight')
plt.close()
print(f"Saved {OUT_DIR}fig_alphamissense_channels.png/pdf")


# ============================================================
# Fig C: Scale amplification of tokenization effect
# ============================================================

fig, ax = plt.subplots(figsize=(7, 5))

pretrain_sizes = [4842, 4842, 114119]
labels = ['v1 (4.8K, 10ep)', 'v3a (4.8K, 100ep)', 'v3b (114K, 30ep)']
deltas_mlp = [9.2, 11.4, 14.5]
deltas_lr = [-1.0, 2.3, 3.1]

ax.plot(range(len(labels)), deltas_mlp, 'o-', color='#2166ac', linewidth=2, markersize=10, label='MLP Δ (codon − char)', zorder=3)
ax.plot(range(len(labels)), deltas_lr, 's--', color='#b2182b', linewidth=1.5, markersize=8, label='LR Δ (codon − char)', zorder=3)

for i, (d_mlp, d_lr) in enumerate(zip(deltas_mlp, deltas_lr)):
    ax.annotate(f'+{d_mlp:.1f}pp', xy=(i, d_mlp), xytext=(i+0.15, d_mlp+0.8),
               fontsize=10, fontweight='bold', color='#2166ac')
    ax.annotate(f'{d_lr:+.1f}pp', xy=(i, d_lr), xytext=(i+0.15, d_lr-1.5),
               fontsize=9, color='#b2182b')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylabel('Δ AUC (codon − character) in percentage points')
ax.set_title('Scale Amplification of Tokenization Effect\n(Task 2: Synonymous Pathogenicity)')
ax.legend(loc='upper left')

ax.fill_between(range(len(labels)), deltas_lr, deltas_mlp, alpha=0.1, color='#2166ac')
ax.text(1, 5, 'Probing-depth\ndependency', ha='center', fontsize=9,
       color='#666666', fontstyle='italic')

plt.tight_layout()
plt.savefig(OUT_DIR + 'fig_scale_amplification.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_DIR + 'fig_scale_amplification.pdf', bbox_inches='tight')
plt.close()
print(f"Saved {OUT_DIR}fig_scale_amplification.png/pdf")

print("\nAll V15 new figures generated!")