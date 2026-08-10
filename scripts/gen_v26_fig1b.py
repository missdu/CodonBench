"""Fig 1b: CodonBench framework flow: 21 models → Four-tier probing → Tasks → Conclusions."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')
os.makedirs(FIG_DIR, exist_ok=True)

BLACK = '#2D2D2D'
GREY = '#999999'
WHITE = '#FFFFFF'
BLUE = '#2166AC'
ORANGE = '#E8833A'
GREEN = '#5AAE61'
PURPLE = '#7B3294'
LIGHT_BLUE = '#D1E5F0'
LIGHT_ORANGE = '#FDE0C5'
LIGHT_GREEN = '#E8F5E9'
LIGHT_PURPLE = '#F3E5F5'
TEAL = '#1A8A7A'
STEP_COLORS = ['#BFD3E6', '#92C5DE', '#4393C3', '#2166AC']


def make_fig1b():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0.3, 12.5)
    ax.set_ylim(0.3, 7.3)
    ax.axis('off')

    # === LEFT: 21 Models ===
    input_x = 1.6
    input_y_top = 6.0
    box_w, box_h = 2.4, 0.7

    ax.text(input_x + box_w / 2, 7.0, '21 Models', fontsize=14, ha='center', va='center',
            fontweight='bold', color=BLACK)

    input_items = [
        ('cLMs (codon)', '#4393C3', '9 models'),
        ('CDS LMs (char)', '#A6D854', '2 models'),
        ('DNA LMs (BPE)', '#F4A582', '2 models'),
        ('pLMs (amino-acid)', '#D6604D', '2 models'),
        ('Traditional', '#999999', '6 baselines'),
    ]

    for i, (label, color, n) in enumerate(input_items):
        y = input_y_top - i * (box_h + 0.12)
        rect = mpatches.FancyBboxPatch((input_x, y - box_h / 2), box_w, box_h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor=BLACK,
                                        linewidth=0.8, alpha=0.8)
        ax.add_patch(rect)
        text_color = 'white' if color != '#A6D854' else BLACK
        ax.text(input_x + box_w / 2, y + 0.05, label, fontsize=11, ha='center', va='center',
                color=text_color, fontweight='bold')
        ax.text(input_x + box_w / 2, y - 0.18, n, fontsize=9, ha='center', va='center',
                color=text_color, alpha=0.85)

    # === MIDDLE: Four-tier probing (staircase) — centered ===
    stair_x_start = 4.5
    stair_y_base = 2.5
    stair_w = 1.15
    stair_h = 0.65
    stair_gap_x = 0.2
    stair_gap_y = 0.55

    stair_center_x = stair_x_start + 1.5 * (stair_w + stair_gap_x)
    ax.text(stair_center_x, 7.0, '4-tier probing', fontsize=14, ha='center', va='center',
            fontweight='bold', color=BLACK)

    tiers = [
        'Zero-shot\nLLR',
        'Linear\n(LR)',
        'Nonlinear\n(MLP)',
        'LoRA\nfine-tune',
    ]

    for i, name in enumerate(tiers):
        x = stair_x_start + i * (stair_w + stair_gap_x)
        y = stair_y_base + i * (stair_h + stair_gap_y - 0.3)
        rect = mpatches.FancyBboxPatch((x, y), stair_w, stair_h,
                                        boxstyle="round,pad=0.06",
                                        facecolor=STEP_COLORS[i], edgecolor=BLACK,
                                        linewidth=0.8, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + stair_w / 2, y + stair_h / 2, name, fontsize=11,
                ha='center', va='center', fontweight='bold',
                color='white' if i >= 2 else BLACK)

    # Depth annotation — diagonal arrow parallel to staircase, offset above
    stair_top_y = stair_y_base + 3 * (stair_h + stair_gap_y - 0.3) + stair_h
    stair_left_x = stair_x_start
    stair_right_x = stair_x_start + 3 * (stair_w + stair_gap_x) + stair_w

    # Offset the arrow above the staircase (perpendicular offset)
    dx = stair_right_x - stair_left_x
    dy = stair_top_y - stair_y_base
    length = np.sqrt(dx**2 + dy**2)
    perp_x = -dy / length * 1.0
    perp_y = dx / length * 1.0

    # Shorten arrow by 1/3: trim from both ends
    trim = 1.0 / 6.0
    ax.annotate('', xy=(stair_right_x + perp_x - dx * trim, stair_top_y + perp_y - dy * trim),
                xytext=(stair_left_x + perp_x + dx * trim, stair_y_base + perp_y + dy * trim),
                arrowprops=dict(arrowstyle='->', color=GREEN, linewidth=2))
    mid_x = (stair_left_x + stair_right_x) / 2 + perp_x
    mid_y = (stair_y_base + stair_top_y) / 2 + perp_y
    angle = np.degrees(np.arctan2(dy, dx))
    ax.text(mid_x, mid_y + perp_y * 0.3,
            'Each tier reveals deeper signal', fontsize=9, ha='center', va='center',
            color=GREEN, fontstyle='italic', rotation=angle)

    # === RIGHT: Six tasks ===
    task_x = 10.0
    task_y_top = 6.2
    task_w, task_h = 2.0, 0.6

    ax.text(task_x + task_w / 2, 7.0, '6 tasks', fontsize=14, ha='center', va='center',
            fontweight='bold', color=BLACK)

    tasks = [
        ('MisPath', 'n=5,000', BLUE, 0.3),
        ('SynPath', 'n=2,840', ORANGE, 0.3),
        ('mRFPExpr', 'n=1,459', TEAL, 0.2),
        ('EcoliExpr', 'n=3,000', TEAL, 0.2),
        ('mRNAStab', 'n=5,000', TEAL, 0.2),
        ('FungalExpr', 'n=7,089', TEAL, 0.2),
    ]

    for i, (name, n, color, alpha) in enumerate(tasks):
        y = task_y_top - i * (task_h + 0.1)
        rect = mpatches.FancyBboxPatch((task_x, y - task_h / 2), task_w, task_h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor=BLACK,
                                        linewidth=0.6, alpha=alpha)
        ax.add_patch(rect)
        ax.text(task_x + task_w / 2, y + 0.06, name, fontsize=11, ha='center', va='center',
                fontweight='bold', color=color)
        ax.text(task_x + task_w / 2, y - 0.14, n, fontsize=9, ha='center', va='center',
                color=GREY)

    # === BOTTOM: Three conclusions ===
    conc_y = 1.2
    conc_w, conc_h = 2.8, 0.8
    conc_gap = 0.5
    conc_x_start = 2.0

    conclusions = [
        ('Channel\nseparation?', PURPLE, LIGHT_PURPLE),
        ('Encoding\ndepth?', BLUE, LIGHT_BLUE),
        ('Channel\nspecificity?', GREEN, LIGHT_GREEN),
    ]

    for i, (text, color, bg) in enumerate(conclusions):
        x = conc_x_start + i * (conc_w + conc_gap)
        rect = mpatches.FancyBboxPatch((x, conc_y - conc_h / 2), conc_w, conc_h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=bg, edgecolor=color,
                                        linewidth=1.2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + conc_w / 2, conc_y, text, fontsize=12, ha='center', va='center',
                fontweight='bold', color=color)

    # Panel label
    ax.text(0.1, 7.3, 'b', fontsize=16, fontweight='bold', va='bottom', color=BLACK)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1b_flow.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig1b_flow.png")


if __name__ == '__main__':
    make_fig1b()
