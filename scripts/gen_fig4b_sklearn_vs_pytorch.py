import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\V36'

models = ['CodonBERT', 'CB-HF', 'EC-80M', 'mRNABERT', 'ESM-2', 'ESM-1b']
sklearn_gain = [4.7, 0.5, 0.7, -2.2, 2.1, -3.7]
pytorch_gain = [-0.0, 0.3, 3.1, -3.0, -0.4, -3.9]

colors_sklearn = '#2166AC'
colors_pytorch = '#D6604D'

fig, ax = plt.subplots(figsize=(6, 3.5))

x = np.arange(len(models))
w = 0.35

bars_sk = ax.bar(x - w/2, sklearn_gain, w, color=colors_sklearn, edgecolor='black', linewidth=0.5, label='sklearn MLP (original)')
bars_pt = ax.bar(x + w/2, pytorch_gain, w, color=colors_pytorch, edgecolor='black', linewidth=0.5, label='PyTorch MLP (unified)')

for i, (sk, pt) in enumerate(zip(sklearn_gain, pytorch_gain)):
    sk_sign = '+' if sk > 0 else ''
    pt_sign = '+' if pt > 0 else ''
    y_sk = sk + 0.3 if sk >= 0 else sk - 0.6
    y_pt = pt + 0.3 if pt >= 0 else pt - 0.6
    ax.text(i - w/2, y_sk, f'{sk_sign}{sk:.1f}', ha='center', fontsize=6.5, fontweight='bold', color=colors_sklearn)
    ax.text(i + w/2, y_pt, f'{pt_sign}{pt:.1f}', ha='center', fontsize=6.5, fontweight='bold', color=colors_pytorch)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=8)
ax.set_ylabel('LOGO-CV gain (MLP − LR, pp)', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.legend(fontsize=7, loc='upper right')
ax.set_title('b  Probe implementation shifts LOGO-CV gain', fontsize=10, fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(-6, 7)

plt.tight_layout()
plt.savefig(f'{OUT}/fig4b_sklearn_vs_pytorch.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig4b_sklearn_vs_pytorch.pdf', bbox_inches='tight')
print(f'Saved to {OUT}/fig4b_sklearn_vs_pytorch.png/pdf')