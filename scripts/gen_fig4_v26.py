import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
plt.subplots_adjust(wspace=0.35, hspace=0.45)

# (a) Pooling ablation — was original a
ax = axes[0, 0]
poolings = ['CLS', 'Mean', 'Variant-pos']
lr_vals = [0.677, 0.693, 0.844]
mlp_vals = [0.814, 0.623, 0.912]
x = np.arange(len(poolings))
w = 0.35
bars_lr = ax.bar(x - w/2, lr_vals, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='LR')
bars_mlp = ax.bar(x + w/2, mlp_vals, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')
for i, (lv, mv) in enumerate(zip(lr_vals, mlp_vals)):
    gain = mv - lv
    sign = '+' if gain > 0 else ''
    ax.text(i, max(lv, mv) + 0.015, f'{sign}{gain*100:.1f} pp', ha='center', fontsize=7, fontweight='bold',
            color='#2166AC' if gain > 0 else '#B2182B')
ax.set_xticks(x)
ax.set_xticklabels(poolings, fontsize=8)
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.50, 1.0)
ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.legend(fontsize=7, loc='upper left')
ax.set_title('a  Pooling ablation (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) mRNABERT comparison — was original f
ax = axes[1, 0]
comp_models = ['CodonBERT', 'CB-HF', 'EC-80M', 'EC-620M', 'CodonTrans.', 'mRNABERT']
comp_lr = [0.734, 0.705, 0.686, 0.785, 0.713, 0.624]
comp_mlp = [0.849, 0.816, 0.812, 0.800, 0.832, 0.667]
comp_gain = [m - l for l, m in zip(comp_lr, comp_mlp)]
x = np.arange(len(comp_models))
colors_gain = ['#2166AC'] * 5 + ['#E8833A']
ax.bar(x, [g * 100 for g in comp_gain], color=colors_gain, edgecolor='black', linewidth=0.5)
for i, g in enumerate(comp_gain):
    ax.text(i, g * 100 + 0.5, f'+{g*100:.1f}' if g > 0 else f'{g*100:.1f}', ha='center', fontsize=6.5, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comp_models, fontsize=6.5)
ax.set_ylabel('LR→MLP gain (pp)', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.set_title('b  LR→MLP gain (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) LoRA SynPath — was original b
ax = axes[0, 1]
models = ['CodonBERT', 'CB-HF', 'EC-80M', 'EC-620M', 'CodonTrans.']
lr_syn = [0.716, 0.668, 0.687, 0.785, 0.713]
mlp_syn = [0.849, 0.816, 0.812, 0.800, 0.832]
lora_syn = [0.912, 0.880, 0.893, 0.880, 0.943]
x = np.arange(len(models))
w = 0.25
ax.bar(x - w, lr_syn, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='LR')
ax.bar(x, mlp_syn, w, color='#4393C3', edgecolor='black', linewidth=0.5, label='MLP')
ax.bar(x + w, lora_syn, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='LoRA')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=6.5)
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.50, 1.0)
ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.legend(fontsize=7, loc='upper left')
ax.set_title('c  LoRA (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (d) Oracle ceiling — was original d
ax = axes[1, 1]
oracle_models = ['Oracle', 'CodonTrans.', 'CodonBERT', 'EC-80M', 'CB-HF', 'EC-620M']
oracle_aucs = [0.949, 0.943, 0.912, 0.893, 0.880, 0.880]
oracle_colors = ['#4CAF50', '#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1']
bars = ax.bar(oracle_models, oracle_aucs, color=oracle_colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=0.949, color='#4CAF50', linestyle='--', linewidth=1.5, label='Oracle (0.949)')
for i, v in enumerate(oracle_aucs):
    if i == 0:
        continue
    pct = v / 0.949 * 100
    ax.text(i, v + 0.005, f'{v:.3f}\n({pct:.1f}%)', ha='center', fontsize=6, fontweight='bold')
ax.text(0, 0.949 + 0.005, '0.949', ha='center', fontsize=6, fontweight='bold', color='#4CAF50')
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.80, 1.0)
ax.legend(fontsize=7, loc='lower right')
ax.set_title('d  Oracle ceiling (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=6.5)

# (e) LoRA MisPath — was original c
ax = axes[0, 2]
lr_mis = [0.662, 0.646, 0.636, 0.617, 0.694]
mlp_mis = [0.677, 0.650, 0.674, 0.632, 0.706]
lora_mis = [0.808, 0.814, 0.696, 0.726, 0.799]
ax.bar(x - w, lr_mis, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='LR')
ax.bar(x, mlp_mis, w, color='#4393C3', edgecolor='black', linewidth=0.5, label='MLP')
ax.bar(x + w, lora_mis, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='LoRA')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=6.5)
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.50, 1.0)
ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.legend(fontsize=7, loc='upper left')
ax.set_title('e  LoRA (MisPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (f) LoRA rank ablation — was original e
ax = axes[1, 2]
ranks = [1, 4, 8, 16, 32, 64]
cb_rank = [0.9053, 0.9023, 0.9121, 0.9109, 0.9186, 0.9158]
ax.plot(ranks, cb_rank, 'o-', color='#2166AC', linewidth=2, markersize=6, label='CodonBERT')
ax.axhline(y=0.9121, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('LoRA rank (r)', fontsize=9)
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.85, 0.95)
ax.set_xticks(ranks)
ax.legend(fontsize=7)
ax.set_title('f  LoRA rank ablation (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(f'{OUT}/fig4_architecture_lora.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig4_architecture_lora.pdf', dpi=300, bbox_inches='tight')
plt.close()
print('Done: fig4_architecture_lora (6 panels)')
