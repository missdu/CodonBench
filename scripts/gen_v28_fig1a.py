"""Fig 1a (V28 revised): Tokenization x Channel access schematic.
Row order: Amino-acid -> Character -> BPE -> Codon (sigma accessibility increasing).
No SynPath LR annotations. No sigma-arrow.
Colors unified with Fig 1c forest plot.
No panel label.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')
os.makedirs(FIG_DIR, exist_ok=True)

PROTEIN = '#D6604D'
CHAR = '#A6D854'
BPE = '#F4A582'
CODON = '#4393C3'
LIGHT_PROTEIN = '#F4C4B8'
LIGHT_CHAR = '#E8F5D0'
LIGHT_BPE = '#FDE0C5'
LIGHT_CODON = '#D1E5F0'
BLACK = '#2D2D2D'
GREY = '#BBBBBB'
WHITE = '#FFFFFF'
RED = '#D6604D'


def make_fig1a():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0.3, 9.5)
    ax.set_ylim(0.3, 6.0)
    ax.axis('off')

    col_y = 5.3
    ax.text(3.8, col_y, 'Amino-acid\nchannel (A)', fontsize=12, ha='center', va='center',
            fontweight='bold', color='#2166AC')
    ax.text(7.0, col_y, 'Synonymous-choice\nchannel (\u03c3)', fontsize=12, ha='center', va='center',
            fontweight='bold', color='#E8833A')

    ax.plot([1.8, 9.0], [4.9, 4.9], color=BLACK, linewidth=0.8)

    tok_data = [
        ('Amino-acid', '(vocab = 20)', 'observed', 'discarded', PROTEIN, LIGHT_PROTEIN, 'e.g., ESM-2'),
        ('Character', '(vocab = 9)', 'dispersed_r', 'dispersed_h', CHAR, LIGHT_CHAR, 'e.g., cdsBERT'),
        ('BPE', '(vocab = 4,096)', 'recoverable', 'partial', BPE, LIGHT_BPE, 'e.g., NT-v2'),
        ('Codon', '(vocab = 69)', 'observed', 'observed', CODON, LIGHT_CODON, 'e.g., CodonBERT'),
    ]

    row_height = 1.05
    row_start_y = 0.7

    for i, (tok_name, vocab_text, a_state, s_state, row_color, row_light, example) in enumerate(tok_data):
        y = row_start_y + i * row_height
        y_center = y + row_height / 2

        ax.text(1.8, y_center + 0.08, tok_name, fontsize=11, ha='center', va='center',
                fontweight='bold', color=BLACK)
        ax.text(1.8, y_center - 0.18, vocab_text, fontsize=9, ha='center', va='center',
                color=GREY)
        ax.text(1.8, y_center - 0.38, example, fontsize=8, ha='center', va='center',
                color=GREY, fontstyle='italic')

        if i > 0:
            ax.plot([1.8, 9.0], [y, y], color=GREY, linewidth=0.4, linestyle='-')

        draw_cell(ax, 3.8, y_center, a_state, '#2166AC', '#D1E5F0')
        draw_cell(ax, 7.0, y_center, s_state, '#E8833A', '#FDE0C5')

    ax.annotate('', xy=(1.0, row_start_y + 4 * row_height), xytext=(1.0, row_start_y),
                arrowprops=dict(arrowstyle='-', color=BLACK, linewidth=1.2))
    ax.text(0.55, row_start_y + 2 * row_height, 'Tokenization', fontsize=11, ha='center',
            va='center', rotation=90, fontweight='bold', color=BLACK)

    ax.annotate('', xy=(9.0, row_start_y + 4 * row_height), xytext=(9.0, row_start_y),
                arrowprops=dict(arrowstyle='->', color='#E8833A', linewidth=1.5))
    ax.text(9.5, row_start_y + 2 * row_height, '\u03c3 access increases', fontsize=11, ha='center',
            va='center', rotation=90, color='#E8833A', fontstyle='italic')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1a_schematic.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig1a_schematic.png")


def draw_cell(ax, cx, cy, state, color, light_color):
    w, h = 2.4, 0.72

    if state == 'observed':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor=color,
                                        linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(cx, cy, 'Observed', fontsize=11, ha='center', va='center',
                color='white', fontweight='bold')

    elif state == 'discarded':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=GREY,
                                        linewidth=1, alpha=0.5, linestyle='--')
        ax.add_patch(rect)
        ax.text(cx, cy + 0.08, 'Discarded', fontsize=11, ha='center', va='center',
                color=RED, fontweight='bold')
        ax.plot([cx - 0.15, cx + 0.15], [cy - 0.15, cy - 0.35], color=RED, linewidth=1.5)
        ax.plot([cx - 0.15, cx + 0.15], [cy - 0.35, cy - 0.15], color=RED, linewidth=1.5)

    elif state == 'recoverable':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=color,
                                        linewidth=1, alpha=0.7)
        ax.add_patch(rect)
        ax.text(cx, cy, 'Recoverable', fontsize=11, ha='center', va='center',
                color=color, fontweight='bold')

    elif state == 'partial':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=color,
                                        linewidth=1, alpha=0.6)
        ax.add_patch(rect)
        ax.text(cx, cy, 'Partial\n(6-mer)', fontsize=10, ha='center', va='center',
                color=color, fontweight='bold')

    elif state == 'dispersed_r':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=color,
                                        linewidth=1, alpha=0.5)
        ax.add_patch(rect)
        ax.text(cx, cy, 'Dispersed\n(recoverable)', fontsize=10, ha='center', va='center',
                color=color, fontweight='bold')

    elif state == 'dispersed_h':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=GREY,
                                        linewidth=1, alpha=0.4, linestyle='--')
        ax.add_patch(rect)
        ax.text(cx, cy, 'Dispersed\n(nonlinear only)', fontsize=9, ha='center', va='center',
                color='#888888', fontweight='bold')


if __name__ == '__main__':
    make_fig1a()
