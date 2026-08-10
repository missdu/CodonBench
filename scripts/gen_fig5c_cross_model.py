import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\V38'
DATA = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results\unified_eval\eval_results\m0_m5_cross_model_results.json'

d = json.load(open(DATA))

combos = ['codonbert_synpath', 'encodon-80m_synpath', 'encodon-80m_mispath', 'esm2-650m_synpath', 'esm2-650m_mispath']
labels = ['CodonBERT\nSynPath', 'EnCodon-80M\nSynPath', 'EnCodon-80M\nMisPath', 'ESM-2-650M\nSynPath', 'ESM-2-650M\nMisPath']

m0_gains = [d[c]['M0_baseline']['gain_perfold_pp'] for c in combos]
m5_gains = [d[c]['M5_sklearn_anchor']['gain_perfold_pp'] for c in combos]
gaps = [m5 - m0 for m5, m0 in zip(m5_gains, m0_gains)]

fig, ax = plt.subplots(figsize=(7.5, 4))

x = np.arange(len(combos))
w = 0.3

bars_m0 = ax.bar(x - w/2, m0_gains, w, color='#D6604D',
                  edgecolor='black', linewidth=0.5, label='M0 (PyTorch baseline)')
bars_m5 = ax.bar(x + w/2, m5_gains, w, color='#2166AC',
                  edgecolor='black', linewidth=0.5, label='M5 (sklearn anchor)')

for i, (m0, m5, gap) in enumerate(zip(m0_gains, m5_gains, gaps)):
    m0_sign = '+' if m0 > 0 else ''
    m5_sign = '+' if m5 > 0 else ''
    gap_sign = '+' if gap > 0 else ''
    y_m0 = m0 + 0.2 if m0 >= 0 else m0 - 0.4
    y_m5 = m5 + 0.2 if m5 >= 0 else m5 - 0.4
    ax.text(i - w/2, y_m0, f'{m0_sign}{m0:.1f}', ha='center', fontsize=7,
            fontweight='bold', color='#D6604D')
    ax.text(i + w/2, y_m5, f'{m5_sign}{m5:.1f}', ha='center', fontsize=7,
            fontweight='bold', color='#2166AC')
    gap_color = '#B2182B' if gap > 0 else '#2166AC'
    ax.text(i, max(m0, m5) + 0.8, f'gap: {gap_sign}{gap:.1f}',
            ha='center', fontsize=7, fontweight='bold', color=gap_color)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('LOGO-CV gain (MLP − LR, pp)', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8)
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(-2.5, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('d  M0→M5 gap is model-conditional, not universal',
             fontsize=10, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5d_cross_model.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig5d_cross_model.pdf', bbox_inches='tight')
print(f'Saved fig5d to {OUT}')
