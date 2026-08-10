"""Fig 1a: Tokenization × Channel access schematic.
Top: I(CDS;Y) = I(A;Y) + I(σ;Y|A)
Below: 4 tokenization rows × 2 channel columns, with visual access states.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE = '#2166AC'
ORANGE = '#E8833A'
LIGHT_BLUE = '#D1E5F0'
LIGHT_ORANGE = '#FDE0C5'
BLACK = '#2D2D2D'
GREY = '#BBBBBB'
WHITE = '#FFFFFF'
GREEN = '#5AAE61'
RED = '#D6604D'

def make_fig1a():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0.3, 9.5)
    ax.set_ylim(0.3, 5.8)
    ax.axis('off')

    # Column headers
    col_y = 5.2
    ax.text(4.0, col_y, 'Amino-acid\nchannel (A)', fontsize=11, ha='center', va='center',
            fontweight='bold', color=BLUE)
    ax.text(7.0, col_y, 'Synonymous-choice\nchannel (σ)', fontsize=11, ha='center', va='center',
            fontweight='bold', color=ORANGE)

    # Separator line
    ax.plot([1.5, 8.5], [4.85, 4.85], color=BLACK, linewidth=0.8)

    # Tokenization rows (top to bottom: amino-acid → codon → BPE → char)
    # List is bottom-to-top (drawn upward), so reverse the visual order
    tok_data = [
        ('Character\n(vocab = 9)', 'dispersed_r', 'dispersed_h', 11),
        ('BPE\n(vocab = 4,096)', 'recoverable', 'partial', 10),
        ('Codon\n(vocab = 69)', 'observed', 'observed', 11),
        ('Amino-acid\n(vocab = 20)', 'observed', 'discarded', 11),
    ]

    row_height = 1.0
    row_start_y = 0.8

    for i, (tok_name, a_state, s_state, tok_fs) in enumerate(tok_data):
        y = row_start_y + i * row_height
        y_center = y + row_height / 2

        # Row label
        ax.text(1.8, y_center, tok_name, fontsize=tok_fs, ha='center', va='center',
                fontweight='bold', color=BLACK)

        # Row separator
        if i > 0:
            ax.plot([1.5, 8.8], [y, y], color=GREY, linewidth=0.4, linestyle='-')

        # A channel cell
        draw_cell(ax, 4.0, y_center, a_state, BLUE, LIGHT_BLUE)

        # σ channel cell
        draw_cell(ax, 7.0, y_center, s_state, ORANGE, LIGHT_ORANGE)

    # Left bracket for row labels
    ax.annotate('', xy=(1.0, row_start_y + 4 * row_height), xytext=(1.0, row_start_y),
                arrowprops=dict(arrowstyle='-', color=BLACK, linewidth=1))
    ax.text(0.6, row_start_y + 2 * row_height, 'Tokenization', fontsize=11, ha='center',
            va='center', rotation=90, fontweight='bold', color=BLACK)

    # Panel label
    ax.text(0.1, 5.8, 'a', fontsize=14, fontweight='bold', va='bottom', color=BLACK)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1a_schematic.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig1a_schematic.png")


def draw_cell(ax, cx, cy, state, color, light_color):
    """Draw a cell indicating channel access state."""
    w, h = 2.4, 0.7

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
        ax.text(cx, cy, 'Discarded', fontsize=11, ha='center', va='center',
                color=RED, fontweight='bold')
        # X mark
        ax.plot([cx - 0.15, cx + 0.15], [cy + 0.18, cy - 0.18], color=RED, linewidth=1.5)
        ax.plot([cx - 0.15, cx + 0.15], [cy - 0.18, cy + 0.18], color=RED, linewidth=1.5)

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
        ax.text(cx, cy, 'Partial\n(6-mer)', fontsize=11, ha='center', va='center',
                color=color, fontweight='bold')

    elif state == 'dispersed_r':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=color,
                                        linewidth=1, alpha=0.5)
        ax.add_patch(rect)
        ax.text(cx, cy, 'Dispersed\n(recoverable)', fontsize=11, ha='center', va='center',
                color=color, fontweight='bold')

    elif state == 'dispersed_h':
        rect = mpatches.FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=light_color, edgecolor=GREY,
                                        linewidth=1, alpha=0.4, linestyle='--')
        ax.add_patch(rect)
        ax.text(cx, cy, 'Dispersed\n(hard to recover)', fontsize=11, ha='center', va='center',
                color='#888888', fontweight='bold')


if __name__ == '__main__':
    make_fig1a()