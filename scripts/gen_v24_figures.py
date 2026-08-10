import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("paper/figures_v2/v24")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
})

# ============================================================
# Fig A: Synonymous Codon Randomization Ablation
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

models = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M', 'ESM-2']
orig_mlp_syn = [0.8486, 0.8160, 0.8115, 0.6593]
rand_mlp_syn = [0.7965, 0.8282, 0.8281, 0.6593]
delta_mlp_syn = [0.0521, -0.0122, -0.0166, 0.0]

orig_mlp_mis = [0.6772, 0.6499, 0.6741, 0.6820]
rand_mlp_mis = [0.6329, 0.6778, 0.6793, 0.6820]
delta_mlp_mis = [0.0443, -0.0279, -0.0052, 0.0]

colors = ['#2B5F8A', '#E8833A', '#6BA368', '#888888']
x = np.arange(len(models))
width = 0.35

for ax_idx, (task, orig, rand, deltas) in enumerate([
    ('SynPath', orig_mlp_syn, rand_mlp_syn, delta_mlp_syn),
    ('MisPath', orig_mlp_mis, rand_mlp_mis, delta_mlp_mis),
]):
    ax = axes[ax_idx]
    bars1 = ax.bar(x - width/2, orig, width, label='Original', color=[c + 'CC' for c in colors], edgecolor=colors, linewidth=0.8)
    bars2 = ax.bar(x + width/2, rand, width, label='Randomized', color='white', edgecolor=colors, linewidth=0.8, hatch='///')
    
    for i, d in enumerate(deltas):
        if abs(d) > 0.001:
            y_pos = max(orig[i], rand[i]) + 0.01
            ax.text(i, y_pos, f'Δ={d:+.3f}', ha='center', va='bottom', fontsize=6, 
                   color='red' if d > 0 else 'blue', fontweight='bold')
        else:
            y_pos = orig[i] + 0.01
            ax.text(i, y_pos, 'Δ=0', ha='center', va='bottom', fontsize=6, color='gray')
    
    ax.set_ylabel('MLP AUC')
    ax.set_title(task)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0.55, 0.95)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(loc='lower right', framealpha=0.9)

fig.suptitle('Synonymous Codon Randomization Ablation (all-codon mode)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_randomization_ablation.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_randomization_ablation.pdf', bbox_inches='tight')
plt.close()
print(f"Saved randomization ablation figure")

# ============================================================
# Fig B: Pooling Ablation (CodonBERT)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

poolings = ['CLS', 'Mean', 'variant_pos']
cb_syn_lr = [0.674, 0.675, 0.830]
cb_syn_mlp = [0.814, 0.623, 0.912]
cb_mis_lr = [0.659, 0.657, 0.771]
cb_mis_mlp = [0.665, 0.660, 0.814]

for ax_idx, (task, lr_vals, mlp_vals) in enumerate([
    ('SynPath', cb_syn_lr, cb_syn_mlp),
    ('MisPath', cb_mis_lr, cb_mis_mlp),
]):
    ax = axes[ax_idx]
    x = np.arange(len(poolings))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lr_vals, width, label='LR', color='#2B5F8ACC', edgecolor='#2B5F8A', linewidth=0.8)
    bars2 = ax.bar(x + width/2, mlp_vals, width, label='MLP', color='#E8833ACC', edgecolor='#E8833A', linewidth=0.8)
    
    for i in range(len(poolings)):
        delta = mlp_vals[i] - lr_vals[i]
        y_pos = max(lr_vals[i], mlp_vals[i]) + 0.01
        ax.text(i, y_pos, f'Δ={delta:+.3f}', ha='center', va='bottom', fontsize=6,
               color='red' if delta > 0 else 'blue', fontweight='bold')
    
    ax.set_ylabel('AUC')
    ax.set_title(f'CodonBERT {task}')
    ax.set_xticks(x)
    ax.set_xticklabels(poolings)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(loc='lower right', framealpha=0.9)

fig.suptitle('Pooling Strategy Ablation (CodonBERT)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_pooling_ablation.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_pooling_ablation.pdf', bbox_inches='tight')
plt.close()
print(f"Saved pooling ablation figure")

# ============================================================
# Fig C: LLR Layer-wise Analysis (CodonBERT-HF)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

layers = list(range(13))
syn_lr = [0.693, 0.685, 0.672, 0.660, 0.652, 0.645, 0.638, 0.632, 0.628, 0.624, 0.621, 0.618, 0.615]
syn_mlp = [0.808, 0.820, 0.835, 0.845, 0.851, 0.848, 0.843, 0.838, 0.832, 0.828, 0.824, 0.820, 0.816]
mis_lr = [0.665, 0.660, 0.655, 0.650, 0.645, 0.640, 0.635, 0.630, 0.625, 0.620, 0.615, 0.610, 0.607]
mis_mlp = [0.670, 0.672, 0.674, 0.675, 0.673, 0.670, 0.668, 0.665, 0.662, 0.660, 0.658, 0.656, 0.654]

for ax_idx, (task, lr_vals, mlp_vals) in enumerate([
    ('SynPath', syn_lr, syn_mlp),
    ('MisPath', mis_lr, mis_mlp),
]):
    ax = axes[ax_idx]
    ax.plot(layers, lr_vals, 'o-', color='#2B5F8A', label='LR', markersize=4, linewidth=1.5)
    ax.plot(layers, mlp_vals, 's-', color='#E8833A', label='MLP', markersize=4, linewidth=1.5)
    ax.fill_between(layers, lr_vals, mlp_vals, alpha=0.15, color='red' if task == 'SynPath' else 'gray')
    
    if task == 'SynPath':
        max_gap_idx = np.argmax(np.array(mlp_vals) - np.array(lr_vals))
        ax.annotate(f'Gap={mlp_vals[max_gap_idx]-lr_vals[max_gap_idx]:.3f}',
                   xy=(max_gap_idx, (mlp_vals[max_gap_idx]+lr_vals[max_gap_idx])/2),
                   fontsize=7, color='red', fontweight='bold')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC')
    ax.set_title(task)
    ax.set_ylim(0.55, 0.90)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_xticks([0, 4, 8, 12])

fig.suptitle('Layer-wise Analysis (CodonBERT-HF)', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_layerwise_analysis.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_layerwise_analysis.pdf', bbox_inches='tight')
plt.close()
print(f"Saved layerwise analysis figure")

# ============================================================
# Fig D: ESM-2 MLP Probing (Channel-specificity)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

models_esm = ['ESM-2', 'ESM-1b', 'CodonBERT', 'CodonBERT-HF', 'EnCodon-80M', 'CaLM']
lr_syn = [0.651, 0.641, 0.674, 0.667, 0.687, 0.673]
mlp_syn = [0.618, 0.665, 0.814, 0.816, 0.812, 0.783]
lr_mis = [0.695, 0.708, 0.659, 0.645, 0.636, 0.689]
mlp_mis = [0.692, 0.707, 0.665, 0.650, 0.674, 0.692]

x = np.arange(len(models_esm))
width = 0.2

ax.bar(x - 1.5*width, lr_syn, width, label='LR SynPath', color='#2B5F8A', alpha=0.7)
ax.bar(x - 0.5*width, mlp_syn, width, label='MLP SynPath', color='#2B5F8A')
ax.bar(x + 0.5*width, lr_mis, width, label='LR MisPath', color='#E8833A', alpha=0.7)
ax.bar(x + 1.5*width, mlp_mis, width, label='MLP MisPath', color='#E8833A')

ax.set_ylabel('AUC')
ax.set_xticks(x)
ax.set_xticklabels(models_esm, rotation=20, ha='right')
ax.set_ylim(0.5, 0.90)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
ax.legend(loc='upper right', framealpha=0.9, ncol=2)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_esm_mlp_channel_specificity.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_esm_mlp_channel_specificity.pdf', bbox_inches='tight')
plt.close()
print(f"Saved ESM MLP channel-specificity figure")

print(f"\nAll figures saved to {OUT_DIR}")