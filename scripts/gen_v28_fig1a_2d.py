"""Two-dimensional channel accessibility plot.
x-axis: A accessibility (conceptual: None / Partial / Full)
y-axis: SynPath LR AUC (actual data)
Gray line: structural gradient (amino-acid concept -> character data -> BPE data -> codon data)
Red dashed: inversion (character data -> amino-acid data)
Only Amino-acid has two points (concept hollow + data solid); others only data points.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')
os.makedirs(FIG_DIR, exist_ok=True)

PROTEIN = '#B71C1C'
CHAR = '#558B2F'
BPE = '#6A1B9A'
CODON = '#1565C0'
BLACK = '#1A1A1A'
GREY = '#666666'
WHITE = '#FFFFFF'
INVERT_C = '#C62828'


def make_accessibility_plot():
    fig, ax = plt.subplots(figsize=(6, 6))

    Y_MIN, Y_MAX = 0.50, 0.90
    X_MIN, X_MAX = -0.15, 1.45

    ax.fill_between([0, X_MAX], Y_MIN, 0.575, alpha=0.12, color='#F7F7F7', zorder=0)
    ax.fill_between([0, X_MAX], 0.575, 0.675, alpha=0.12, color='#D1E5F0', zorder=0)
    ax.fill_between([0, X_MAX], 0.675, 0.775, alpha=0.12, color='#92C5DE', zorder=0)
    ax.fill_between([0, X_MAX], 0.775, Y_MAX, alpha=0.12, color='#4393C3', zorder=0)

    aa_x, aa_y_data, aa_y_concept = 1.0, 0.680, 0.540
    ch_x, ch_y = 0.5, 0.598
    bpe_x, bpe_y = 0.85, 0.699
    cod_x, cod_y = 1.0, 0.785

    ax.plot([aa_x, ch_x, bpe_x, cod_x], [aa_y_concept, ch_y, bpe_y, cod_y],
            color=GREY, linewidth=1.4, linestyle='-', zorder=2, alpha=0.85)

    ax.scatter([aa_x], [aa_y_concept], marker='P', s=220,
               facecolors='none', edgecolors=PROTEIN, linewidths=2.5, zorder=4)

    ax.scatter([aa_x], [aa_y_data], marker='P', s=220,
               color=PROTEIN, edgecolors=BLACK, linewidths=0.8, zorder=5)
    ax.annotate('Amino-acid', xy=(aa_x, aa_y_data),
                xytext=(16, 8), textcoords='offset points',
                fontsize=14, fontweight='bold', color=PROTEIN, ha='left', va='center')
    ax.annotate('(ESM-2)', xy=(aa_x, aa_y_data),
                xytext=(16, -8), textcoords='offset points',
                fontsize=12, color=PROTEIN, ha='left', va='center')

    ax.scatter([ch_x], [ch_y], marker='p', s=200,
               color=CHAR, edgecolors=BLACK, linewidths=0.8, zorder=5)
    ax.annotate('Character', xy=(ch_x, ch_y),
                xytext=(-16, 8), textcoords='offset points',
                fontsize=14, fontweight='bold', color=CHAR, ha='right', va='center')
    ax.annotate('(cdsBERT)', xy=(ch_x, ch_y),
                xytext=(-16, -8), textcoords='offset points',
                fontsize=12, color=CHAR, ha='right', va='center')

    ax.scatter([bpe_x], [bpe_y], marker='^', s=200,
               color=BPE, edgecolors=BLACK, linewidths=0.8, zorder=5)
    ax.annotate('BPE', xy=(bpe_x, bpe_y),
                xytext=(-16, 8), textcoords='offset points',
                fontsize=14, fontweight='bold', color=BPE, ha='right', va='center')
    ax.annotate('(NT-v2)', xy=(bpe_x, bpe_y),
                xytext=(-16, -8), textcoords='offset points',
                fontsize=12, color=BPE, ha='right', va='center')

    ax.scatter([cod_x], [cod_y], marker='o', s=240,
               color=CODON, edgecolors=BLACK, linewidths=0.8, zorder=5)
    ax.annotate('Codon', xy=(cod_x, cod_y),
                xytext=(-16, 8), textcoords='offset points',
                fontsize=14, fontweight='bold', color=CODON, ha='right', va='center')
    ax.annotate('(EnCodon)', xy=(cod_x, cod_y),
                xytext=(-16, -8), textcoords='offset points',
                fontsize=12, color=CODON, ha='right', va='center')

    ax.plot([ch_x, aa_x], [ch_y, aa_y_data],
            color=INVERT_C, linewidth=1.8, linestyle='--', zorder=3, alpha=0.85)

    ax.text(0.85, 0.62, 'inverted', fontsize=11, color=INVERT_C,
            fontstyle='italic', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=INVERT_C, alpha=0.85))

    ax.set_xlabel('A accessibility', fontsize=14, fontweight='bold', color=BLACK)
    ax.set_ylabel('SynPath LR AUC', fontsize=14, fontweight='bold', color=BLACK)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect('auto')

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['None', 'Partial', 'Full'], fontsize=12, color=BLACK)
    ax.set_yticks([0.55, 0.65, 0.75, 0.85])
    ax.set_yticklabels(['0.55', '0.65', '0.75', '0.85'], fontsize=12, color=BLACK)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_color(BLACK)
    ax.tick_params(axis='both', colors=BLACK)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1a_accessibility_2d.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig1a_accessibility_2d.png")


if __name__ == '__main__':
    make_accessibility_plot()
