import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

CC, CH = "#2166AC", "#E08214"

versions = ['v1', 'v3a', 'v3b', 'v4', 'v2']
x = np.arange(len(versions))
w = 0.25

mlp_codon = [0.655, 0.670, 0.674, 0.681, 0.648]
mlp_char  = [0.648, 0.634, 0.702, 0.643, 0.661]

for i in range(len(versions)):
    if mlp_codon[i] is not None:
        ax.bar(x[i] - w/2, mlp_codon[i], w, color=CC, edgecolor='gray', linewidth=0.5,
               label='Codon MLP' if i == 0 else '')
    if mlp_char[i] is not None:
        ax.bar(x[i] + w/2, mlp_char[i], w, color=CH, edgecolor='gray', linewidth=0.5,
               label='Char MLP' if i == 0 else '')

ax.annotate('', xy=(x[2]+w/2, 0.702-0.006), xytext=(x[2]-w/2, 0.674-0.006),
            arrowprops=dict(arrowstyle='<->', color='#C62828', lw=1.5))
ax.text(x[2], 0.674-0.025, 'char > codon\n(reversed)', ha='center', fontsize=8, fontweight='bold', color='#C62828')

ax.annotate('', xy=(x[3]-w/2, 0.681-0.006), xytext=(x[3]+w/2, 0.643-0.006),
            arrowprops=dict(arrowstyle='<->', color=CC, lw=1.5))
ax.text(x[3], 0.643-0.025, 'codon > char\n(+3.8pp)', ha='center', fontsize=8, fontweight='bold', color=CC)

ax.text(x[4], 0.625, 'No gap\n(synthetic)', ha='center', fontsize=7, color='#888', style='italic')

ax.axvline(x=3.5, color='#CCCCCC', linewidth=1, linestyle='--')

ax.text(0.5, 0.76, 'SynPath: codon > char by +9-15 pp (MLP)', 
        ha='center', fontsize=9, color=CC, fontweight='bold', transform=ax.get_xaxis_transform(),
        style='italic')
ax.text(0.5, 0.72, 'MisPath: no codon advantage (reversed at v3b, restored at v4)', 
        ha='center', fontsize=9, color='#C62828', fontweight='bold', transform=ax.get_xaxis_transform(),
        style='italic')

ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=10)
ax.set_ylabel('MisPath AUC (MLP)', fontsize=11)
ax.set_title('MisPath: codon advantage is absent or reversed', fontsize=12, fontweight='bold')
ax.set_ylim(0.58, 0.78)
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_path = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\v26\fig2h_mispath_reversal.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
fig.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print('Saved fig2h_mispath_reversal')
