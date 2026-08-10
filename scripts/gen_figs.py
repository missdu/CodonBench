import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

models = ['EnCodon-80M', 'CodonBERT', 'CodonBERT-HF']

# Best results (200 tx, balanced)
task2_auc = [0.6134, 0.6462, 0.6806]
task2_std = [0.0198, 0.0092, 0.0254]
task3_auc = [0.8241, 0.8204, 0.8498]
task3_std = [0.0489, 0.0461, 0.0361]

# Zero-shot (real CDS)
zs_task2 = [0.508, 0.500, 0.608]
zs_task3 = [0.519, 0.502, 0.501]

# Fig 1: Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(models))
width = 0.35

ax = axes[0]
ax.bar(x - width/2, task2_auc, width, yerr=task2_std, label='Downstream', color='#2196F3', capsize=3, alpha=0.9)
ax.bar(x + width/2, zs_task2, width, label='Zero-shot', color='#FF9800', capsize=3, alpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.set_ylabel('ROC-AUC'); ax.set_title('Task 2: Missense Variant Pathogenicity')
ax.set_xticks(x); ax.set_xticklabels(models, rotation=15); ax.set_ylim(0.4, 1.0); ax.legend(loc='upper left')

ax = axes[1]
ax.bar(x - width/2, task3_auc, width, yerr=task3_std, label='Downstream', color='#2196F3', capsize=3, alpha=0.9)
ax.bar(x + width/2, zs_task3, width, label='Zero-shot', color='#FF9800', capsize=3, alpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.set_ylabel('ROC-AUC'); ax.set_title('Task 3: Synonymous Variant Pathogenicity')
ax.set_xticks(x); ax.set_xticklabels(models, rotation=15); ax.set_ylim(0.4, 1.0); ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('paper/figures/fig1_benchmark_results.png', dpi=300, bbox_inches='tight')
print("Saved fig1")

# Fig 2: Heatmap
fig, ax = plt.subplots(figsize=(8, 5))
data = np.array([
    [0.508, 0.519, 0.613, 0.824],
    [0.500, 0.502, 0.646, 0.820],
    [0.608, 0.501, 0.681, 0.850],
])
im = ax.imshow(data, cmap='RdYlGn', vmin=0.45, vmax=0.9, aspect='auto')
ax.set_xticks(np.arange(4)); ax.set_xticklabels(['ZS Task2', 'ZS Task3', 'DS Task2', 'DS Task3'], rotation=30)
ax.set_yticks(np.arange(3)); ax.set_yticklabels(models)
for i in range(3):
    for j in range(4):
        val = data[i, j]
        color = 'white' if val < 0.55 or val > 0.8 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontweight='bold')
ax.set_title('CodonBench Results: Zero-shot (ZS) vs Downstream (DS)')
fig.colorbar(im, ax=ax, label='ROC-AUC')
plt.tight_layout()
plt.savefig('paper/figures/fig2_results_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved fig2")

# Fig 3: Context comparison
fig, ax = plt.subplots(figsize=(8, 5))
syn_task3 = [0.763, 0.714, 0.674]
real_task3 = [0.824, 0.820, 0.850]
ax.bar(x - width/2, syn_task3, width, label='Synthetic context', color='#9E9E9E', alpha=0.8)
ax.bar(x + width/2, real_task3, width, label='Real CDS context', color='#4CAF50', alpha=0.9)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('ROC-AUC (Task 3: Synonymous)'); ax.set_title('Impact of Real CDS Context on Downstream Evaluation')
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.4, 1.0); ax.legend()
for i in range(3):
    improvement = real_task3[i] - syn_task3[i]
    ax.annotate(f'+{improvement:.1%}', xy=(i + width/2, real_task3[i] + 0.01), ha='center', fontsize=10, color='#4CAF50', fontweight='bold')
plt.tight_layout()
plt.savefig('paper/figures/fig3_context_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig3")

# Fig 4: Task difficulty (full ClinVar)
fig, ax = plt.subplots(figsize=(8, 5))
task2_full = [0.633, 0.660, 0.659]
task3_full = [0.686, 0.734, 0.705]
ax.bar(x - width/2, task2_full, width, label='Task 2: Missense', color='#FF5722', alpha=0.8)
ax.bar(x + width/2, task3_full, width, label='Task 3: Synonymous', color='#3F51B5', alpha=0.8)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('ROC-AUC (Downstream, Full ClinVar)'); ax.set_title('Task Difficulty: Synonymous > Missense for cLMs')
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.4, 1.0); ax.legend()
plt.tight_layout()
plt.savefig('paper/figures/fig4_task_difficulty.png', dpi=300, bbox_inches='tight')
print("Saved fig4")

print("\nAll 4 figures saved to paper/figures/")