"""Combine Fig 1a, 1b, 1c into a single Fig 1.
Layout: top row = a (left) + b (right), bottom row = c (full width).
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')

WHITE = '#FFFFFF'
BLACK = '#2D2D2D'

def combine_fig1():
    img_a = mpimg.imread(os.path.join(FIG_DIR, 'fig1a_schematic.png'))
    img_b = mpimg.imread(os.path.join(FIG_DIR, 'fig1b_flow.png'))
    img_c = mpimg.imread(os.path.join(FIG_DIR, 'fig2_forest_compact.png'))

    fig = plt.figure(figsize=(18, 14), facecolor=WHITE)

    # Top row: a (left, narrower) + b (right, wider)
    # a aspect ~7:4, b aspect ~10:5.5
    ax_a = fig.add_axes([0.02, 0.48, 0.38, 0.50])
    ax_b = fig.add_axes([0.42, 0.48, 0.56, 0.50])

    # Bottom row: c (full width)
    ax_c = fig.add_axes([0.05, 0.03, 0.90, 0.44])

    for ax, img in [(ax_a, img_a), (ax_b, img_b), (ax_c, img_c)]:
        ax.imshow(img)
        ax.axis('off')

    plt.savefig(os.path.join(FIG_DIR, 'fig1_combined.png'),
                dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.savefig(os.path.join(FIG_DIR, 'fig1_combined.pdf'),
                dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig1_combined.png")

if __name__ == '__main__':
    combine_fig1()