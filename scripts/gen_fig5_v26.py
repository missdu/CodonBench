import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26'

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
plt.subplots_adjust(wspace=0.35, hspace=0.45)

models_reg = ['CB-HF', 'CB', 'EC-80M']
colors_reg = ['#08519C', '#4292C6', '#9ECAE1']
x = np.arange(len(models_reg))

# (a) Synonymous codon randomization ablation
ax = axes[0, 0]
models = ['CodonBERT', 'EnCodon-80M', 'ESM-2']
orig_syn_mlp = [0.849, 0.812, 0.618]
rand_syn_mlp = [0.797, 0.828, 0.618]

x_rand = np.arange(len(models))
w = 0.35
ax.bar(x_rand - w/2, orig_syn_mlp, w, color='#4393C3', edgecolor='black', linewidth=0.5, label='Original')
ax.bar(x_rand + w/2, rand_syn_mlp, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='Randomized')
for i in range(len(models)):
    delta = rand_syn_mlp[i] - orig_syn_mlp[i]
    sign = '+' if delta > 0 else ''
    color = '#B2182B' if delta < 0 else '#2166AC'
    ax.text(i, max(orig_syn_mlp[i], rand_syn_mlp[i]) + 0.015, f'{sign}{delta*100:.1f} pp',
            ha='center', fontsize=7, fontweight='bold', color=color)
ax.set_xticks(x_rand)
ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('AUC', fontsize=9)
ax.set_ylim(0.50, 0.95)
ax.axhline(y=0.5, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('a  Codon randomization (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (b) mRFPExpr — Ridge vs MLP
ax = axes[0, 1]
ridge_r2 = [0.253, -0.022, 0.124]
mlp_r2 = [0.463, 0.041, 0.361]
w = 0.3
ax.bar(x - w/2, ridge_r2, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='Ridge')
ax.bar(x + w/2, mlp_r2, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')
for i in range(len(models_reg)):
    delta = (mlp_r2[i] - ridge_r2[i]) * 100
    ax.text(i, max(ridge_r2[i], mlp_r2[i]) + 0.03, f'+{delta:.1f}pp',
            ha='center', fontsize=6.5, fontweight='bold', color='#B2182B')
ax.set_xticks(x)
ax.set_xticklabels(models_reg, fontsize=8)
ax.set_ylabel('R²', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.set_ylim(-0.15, 0.6)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('b  mRFPExpr (within-protein)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (c) EcoliExpr — Ridge vs MLP
ax = axes[0, 2]
ridge_r2 = [-0.397, -0.308, -0.455]
mlp_r2 = [-0.161, -0.106, -0.008]
w = 0.3
ax.bar(x - w/2, ridge_r2, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='Ridge')
ax.bar(x + w/2, mlp_r2, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')
for i in range(len(models_reg)):
    delta = (mlp_r2[i] - ridge_r2[i]) * 100
    y_pos = max(ridge_r2[i], mlp_r2[i]) + 0.02
    ax.text(i, y_pos, f'+{delta:.1f}pp',
            ha='center', fontsize=6.5, fontweight='bold', color='#2166AC')
ax.set_xticks(x)
ax.set_xticklabels(models_reg, fontsize=8)
ax.set_ylabel('R²', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.set_ylim(-0.6, 0.15)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('c  EcoliExpr (cross-protein)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (d) FungalExpr — Ridge vs MLP
ax = axes[1, 0]
ridge_r2 = [0.563, 0.544, 0.483]
mlp_r2 = [0.631, 0.613, 0.643]
w = 0.3
ax.bar(x - w/2, ridge_r2, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='Ridge')
ax.bar(x + w/2, mlp_r2, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')
for i in range(len(models_reg)):
    delta = (mlp_r2[i] - ridge_r2[i]) * 100
    ax.text(i, max(ridge_r2[i], mlp_r2[i]) + 0.02, f'+{delta:.1f}pp',
            ha='center', fontsize=6.5, fontweight='bold', color='#B2182B')
ax.set_xticks(x)
ax.set_xticklabels(models_reg, fontsize=8)
ax.set_ylabel('R²', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.set_ylim(-0.1, 0.75)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('d  FungalExpr (cross-species)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (e) mRNAStab — Ridge vs MLP
ax = axes[1, 1]
ridge_r2 = [0.011, -0.013, -0.151]
mlp_r2 = [0.019, 0.003, 0.031]
w = 0.3
ax.bar(x - w/2, ridge_r2, w, color='#92C5DE', edgecolor='black', linewidth=0.5, label='Ridge')
ax.bar(x + w/2, mlp_r2, w, color='#2166AC', edgecolor='black', linewidth=0.5, label='MLP')
for i in range(len(models_reg)):
    delta = (mlp_r2[i] - ridge_r2[i]) * 100
    y_pos = max(ridge_r2[i], mlp_r2[i]) + 0.02
    ax.text(i, y_pos, f'+{delta:.1f}pp',
            ha='center', fontsize=6.5, fontweight='bold', color='#2166AC')
ax.set_xticks(x)
ax.set_xticklabels(models_reg, fontsize=8)
ax.set_ylabel('R²', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.set_ylim(-0.25, 0.15)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('e  mRNAStab (mRNA stability)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# (f) CKA similarity
ax = axes[1, 2]
pairs = ['EC-620M\nvs EC-80M', 'CodonTrans.\nvs CB', 'CaLM\nvs CB', 'EC-620M\nvs CB-HF', 'EC-620M\nvs CB']
linear_cka = [0.661, 0.438, 0.396, 0.400, 0.308]
rbf_cka = [0.985, 0.983, 0.894, 0.983, 0.983]

x_cka = np.arange(len(pairs))
w = 0.35
ax.bar(x_cka - w/2, linear_cka, w, color='#4393C3', edgecolor='black', linewidth=0.5, label='Linear CKA')
ax.bar(x_cka + w/2, rbf_cka, w, color='#F4A582', edgecolor='black', linewidth=0.5, label='RBF CKA')
ax.set_xticks(x_cka)
ax.set_xticklabels(pairs, fontsize=6.5)
ax.set_ylabel('CKA similarity', fontsize=9)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=7, loc='center right')
ax.set_title('f  CKA similarity (SynPath)', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(f'{OUT}/fig5_signal_regression_cka.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{OUT}/fig5_signal_regression_cka.pdf', bbox_inches='tight')
plt.close()
print('Done: fig5_signal_regression_cka (6 panels)')
