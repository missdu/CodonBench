import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 8

OUT = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\figures_v2\V38'

# ============================================================
# Fig 5a: Waterfall decomposition of +4.7 pp
# ============================================================
steps = ['M0\nbaseline', 'M0→M1\n+seed', 'M1→M2\n+ES', 'M2→M3\n+wd',
         'M3→M4\n+epochs', 'M4→M5\n+sklearn']
increments = [-1.6, 2.5, 0.2, 0.6, 0.0, 3.0]
cumulative = [-1.6, 0.9, 1.1, 1.7, 1.7, 4.7]

fig, ax = plt.subplots(figsize=(6.5, 3.5))

bar_colors = []
for inc in increments:
    if inc < 0:
        bar_colors.append('#4393C3')
    elif inc == 0:
        bar_colors.append('#999999')
    else:
        bar_colors.append('#D6604D')

bars = ax.bar(range(len(steps)), increments, color=bar_colors,
              edgecolor='black', linewidth=0.5, width=0.65)

for i, (inc, cum) in enumerate(zip(increments, cumulative)):
    sign = '+' if inc > 0 else ''
    y_pos = inc + 0.15 if inc >= 0 else inc - 0.35
    ax.text(i, y_pos, f'{sign}{inc:.1f}', ha='center', fontsize=7.5,
            fontweight='bold', color='#333333')

    cum_sign = '+' if cum > 0 else ''
    ax.text(i, -5.2, f'({cum_sign}{cum:.1f})', ha='center', fontsize=6.5,
            color='#666666', style='italic')

ax.plot(range(len(steps)), cumulative, 'ko-', markersize=4, linewidth=1.2,
        zorder=5, label='Cumulative gain')

ax.set_xticks(range(len(steps)))
ax.set_xticklabels(steps, fontsize=7.5)
ax.set_ylabel('Incremental gain (pp)', fontsize=9)
ax.axhline(y=0, color='grey', linewidth=0.8, linestyle='-')
ax.set_ylim(-6.5, 5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=7, loc='upper left')

ax.annotate('', xy=(5, 4.7), xytext=(5, -1.6),
            arrowprops=dict(arrowstyle='<->', color='#B2182B', lw=1.5))
ax.text(5.35, 1.5, '+4.7 pp\n(reported)', fontsize=7, color='#B2182B',
        fontweight='bold', ha='left', va='center')

ax.set_title('a  Waterfall: six innocuous defaults → +4.7 pp',
             fontsize=10, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5a_waterfall.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig5a_waterfall.pdf', bbox_inches='tight')
print(f'Saved fig5a to {OUT}')

# ============================================================
# Fig 5b: Pooling gate sklearn vs PyTorch
# ============================================================
fig2, ax2 = plt.subplots(figsize=(5, 3.5))

pooling = ['CLS', 'Mean']
sklearn_gains = [4.7, 0.6]
pytorch_gains = [0.9, 1.2]
sklearn_p = [0.099, 0.66]
pytorch_p = [0.73, 0.66]

x = np.arange(len(pooling))
w = 0.32

bars_sk = ax2.bar(x - w/2, sklearn_gains, w, color='#2166AC',
                  edgecolor='black', linewidth=0.5, label='sklearn MLP')
bars_pt = ax2.bar(x + w/2, pytorch_gains, w, color='#D6604D',
                  edgecolor='black', linewidth=0.5, label='PyTorch MLP')

for i, (sk, pt, sk_p, pt_p) in enumerate(zip(sklearn_gains, pytorch_gains,
                                               sklearn_p, pytorch_p)):
    sk_sign = '+' if sk > 0 else ''
    pt_sign = '+' if pt > 0 else ''
    y_sk = sk + 0.2 if sk >= 0 else sk - 0.5
    y_pt = pt + 0.2 if pt >= 0 else pt - 0.5
    ns_sk = '' if sk_p < 0.05 else ' n.s.'
    ns_pt = '' if pt_p < 0.05 else ' n.s.'
    ax2.text(i - w/2, y_sk, f'{sk_sign}{sk:.1f}{ns_sk}', ha='center',
             fontsize=7.5, fontweight='bold', color='#2166AC')
    ax2.text(i + w/2, y_pt, f'{pt_sign}{pt:.1f}{ns_pt}', ha='center',
             fontsize=7.5, fontweight='bold', color='#D6604D')

ax2.set_xticks(x)
ax2.set_xticklabels(pooling, fontsize=9)
ax2.set_ylabel('LOGO-CV gain (MLP − LR, pp)', fontsize=9)
ax2.axhline(y=0, color='grey', linewidth=0.8)
ax2.legend(fontsize=7, loc='upper right')
ax2.set_ylim(-1.5, 6.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.annotate('Pooling gate\n(sklearn only)',
             xy=(0 - w/2, 4.7), xytext=(1.0, 5.5),
             fontsize=7, color='#2166AC', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1))

ax2.set_title('c  Pooling "gate" vanishes under matched probe',
              fontsize=10, fontweight='bold', loc='left')

plt.tight_layout()
plt.savefig(f'{OUT}/fig5c_pooling_gate.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig5c_pooling_gate.pdf', bbox_inches='tight')
print(f'Saved fig5c to {OUT}')