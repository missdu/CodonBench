import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'figures_v2', 'V37')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
})

C_DIM1 = '#4393C3'
C_DIM2 = '#D6604D'
C_DIM3 = '#66C2A5'
C_DIM4 = '#FC8D62'
C_DIM5 = '#8DA0CB'

def draw_fig1b_framework():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5, 6.7, 'b  CodonBench: five-dimensional evaluation framework',
            ha='center', va='center', fontsize=10, fontweight='bold')

    models_x, models_y = 1.5, 5.0
    box_m = mpatches.FancyBboxPatch((models_x - 1.0, models_y - 0.5), 2.0, 1.0,
                                     boxstyle="round,pad=0.12", facecolor='#E8E8E8',
                                     edgecolor='#555', linewidth=1.0)
    ax.add_patch(box_m)
    ax.text(models_x, models_y + 0.15, '21 models', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#333')
    ax.text(models_x, models_y - 0.2, '9 cLM · 2 CDS · 2 pLM\n2 DNA · 6 baselines',
            ha='center', va='center', fontsize=5.5, color='#555')

    dims_x = 5.0
    dims_top = 5.8
    dims = [
        ('1. Data split', 'Random / LOGO-CV', C_DIM1),
        ('2. Probe depth', 'LR / MLP / LoRA', C_DIM2),
        ('3. Epoch selection', 'Best-epoch / Validation', C_DIM3),
        ('4. Pooling strategy', 'CLS / Mean', C_DIM4),
        ('5. Probe implementation', 'sklearn / PyTorch', C_DIM5),
    ]

    box_h = 0.55
    box_w = 2.8
    gap = 0.12
    total_h = len(dims) * box_h + (len(dims) - 1) * gap
    start_y = dims_top - total_h / 2

    for i, (label, detail, color) in enumerate(dims):
        y = start_y + (len(dims) - 1 - i) * (box_h + gap)
        box = mpatches.FancyBboxPatch((dims_x - box_w/2, y - box_h/2), box_w, box_h,
                                       boxstyle="round,pad=0.08", facecolor=color,
                                       alpha=0.2, edgecolor=color, linewidth=1.0)
        ax.add_patch(box)
        ax.text(dims_x - box_w/2 + 0.15, y + 0.05, label, ha='left', va='center',
                fontsize=7, fontweight='bold', color=color)
        ax.text(dims_x + box_w/2 - 0.15, y - 0.1, detail, ha='right', va='center',
                fontsize=5.5, color='#444')

    ax.annotate('', xy=(dims_x - box_w/2 - 0.05, start_y + total_h/2),
                xytext=(models_x + 1.05, models_y),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    tasks_x, tasks_y = 8.5, 5.0
    box_t = mpatches.FancyBboxPatch((tasks_x - 1.0, tasks_y - 0.5), 2.0, 1.0,
                                     boxstyle="round,pad=0.12", facecolor='#E8E8E8',
                                     edgecolor='#555', linewidth=1.0)
    ax.add_patch(box_t)
    ax.text(tasks_x, tasks_y + 0.15, '6 tasks', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#333')
    ax.text(tasks_x, tasks_y - 0.2, '2 classification · 4 regression\nSynPath · MisPath · Expr · Stab',
            ha='center', va='center', fontsize=5.5, color='#555')

    ax.annotate('', xy=(tasks_x - 1.05, tasks_y),
                xytext=(dims_x + box_w/2 + 0.05, start_y + total_h/2),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    questions_y = 1.2
    questions = [
        'Q1: Does the codon advantage survive gene-held-out evaluation?',
        'Q2: Is the advantage linear or nonlinear?',
        'Q3: Does validation-based selection change the conclusion?',
        'Q4: Does pooling strategy gate the signal?',
        'Q5: Does probe implementation determine the reported gain?',
    ]
    q_colors = [C_DIM1, C_DIM2, C_DIM3, C_DIM4, C_DIM5]

    ax.text(5, questions_y + 0.6, 'Five questions addressed by the framework',
            ha='center', va='center', fontsize=7.5, fontstyle='italic', color='#444')

    for i, (q, qc) in enumerate(zip(questions, q_colors)):
        y = questions_y + 0.2 - i * 0.3
        ax.text(0.5, y, q, ha='left', va='center', fontsize=6, color=qc, fontweight='bold')

    ax.annotate('', xy=(5, questions_y + 0.35),
                xytext=(5, start_y - box_h/2 - 0.1),
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.8, linestyle='--'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1b_framework.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig1b_framework.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved fig1b_framework to {OUT}')

draw_fig1b_framework()