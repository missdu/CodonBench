"""
Generate Fig 7 (context ablation) and Fig 9 (GC3/CAI interpretability)
for CodonBench v11 paper.
Nature MI style: clean, high DPI, consistent color scheme.
  - protein-channel: blue (#2166AC)
  - synonymous-channel: orange (#E08214)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

BLUE = '#2166AC'
ORANGE = '#E08214'
GRAY = '#888888'
LIGHT_BLUE = '#D1E5F0'
LIGHT_ORANGE = '#FDDDAC'
BLACK = '#1a1a1a'
WHITE = '#ffffff'

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures_v2')
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_json(fname):
    fpath = os.path.join(RESULTS_DIR, fname)
    for enc in ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-8', 'latin-1']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot decode {fname}")


def fig7_context_ablation():
    """
    Fig 7: Context ablation — Real CDS vs Synthetic context.
    Left panel: Downstream LR probing (Task 1 Missense, strict comparison).
    Right panel: Zero-shot LLR (Task 1 Missense, both contexts).
    """
    models = ['EnCodon-80M', 'CodonBERT', 'CodonBERT-HF']
    model_labels = ['EnCodon\n80M', 'CodonBERT', 'CodonBERT\n-HF']

    # Downstream LR Task 1: synthetic vs real CDS (strictly comparable)
    ds_synth = [0.5236, 0.5598, 0.5242]
    ds_synth_std = [0.0188, 0.0169, 0.0059]
    ds_real = [0.6329, 0.6602, 0.6594]
    ds_real_std = [0.0233, 0.0136, 0.0135]

    # Zero-shot Task 1: synthetic vs real CDS
    zs_synth = [0.5033, 0.5218, 0.5112]
    zs_real = [0.5295, 0.4893, 0.6193]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    x = np.arange(len(models))
    width = 0.35

    # Left panel: Downstream LR probing
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, ds_synth, width, yerr=ds_synth_std,
                    label='Synthetic context', color=LIGHT_BLUE,
                    edgecolor=GRAY, linewidth=0.8, capsize=3, error_kw={'linewidth': 0.8})
    bars2 = ax1.bar(x + width/2, ds_real, width, yerr=ds_real_std,
                    label='Real CDS context', color=BLUE,
                    edgecolor=BLACK, linewidth=0.8, capsize=3, error_kw={'linewidth': 0.8})

    for i in range(len(models)):
        delta = ds_real[i] - ds_synth[i]
        pct = delta / ds_synth[i] * 100
        y_max = max(ds_real[i] + ds_real_std[i], ds_synth[i] + ds_synth_std[i])
        ax1.annotate(f'+{pct:.0f}%',
                    xy=(x[i], y_max + 0.015),
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=BLUE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=9)
    ax1.set_ylabel('AUC', fontsize=10)
    ax1.set_title('Downstream LR Probing\n(Task 1: Missense)', fontsize=10, fontweight='bold', pad=8)
    ax1.set_ylim(0.4, 0.85)
    ax1.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=9)
    ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)

    # Right panel: Zero-shot
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, zs_synth, width,
                    label='Synthetic context', color=LIGHT_ORANGE,
                    edgecolor=GRAY, linewidth=0.8)
    bars4 = ax2.bar(x + width/2, zs_real, width,
                    label='Real CDS context', color=ORANGE,
                    edgecolor=BLACK, linewidth=0.8)

    for i in range(len(models)):
        delta = zs_real[i] - zs_synth[i]
        if delta > 0:
            pct = delta / zs_synth[i] * 100
            label = f'+{pct:.0f}%'
        else:
            pct = delta / zs_synth[i] * 100
            label = f'{pct:.0f}%'
        y_max = max(zs_real[i], zs_synth[i]) + 0.02
        ax2.annotate(label,
                    xy=(x[i], y_max),
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold',
                    color=ORANGE if delta > 0 else GRAY)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=9)
    ax2.set_ylabel('AUC', fontsize=10)
    ax2.set_title('Zero-shot LLR\n(Task 1: Missense)', fontsize=10, fontweight='bold', pad=8)
    ax2.set_ylim(0.4, 0.75)
    ax2.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=9)
    ax2.legend(fontsize=8, loc='upper left', framealpha=0.9)

    fig.tight_layout(w_pad=2)

    for ext in ['pdf', 'png']:
        outpath = os.path.join(FIG_DIR, f'fig7_context.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved: {outpath}')
    plt.close(fig)


def fig9_interpretability():
    """
    Fig 9: GC3/CAI interpretability — Embedding attribute regression R².
    Grouped bar chart: 3 cLMs x 4 attributes (GC3, CAI, position, pathogenicity).
    """
    # Data from biological_findings_results.json
    data = {
        'CodonBERT': {
            'GC3': 0.948, 'GC3_std': 0.0064,
            'CAI': 0.5944, 'CAI_std': 0.0221,
            'Position': -0.7261, 'Position_std': 0.1017,
            'Pathogenicity\n(LR AUC)': 0.6881, 'Pathogenicity_std': 0.0181,
        },
        'EnCodon-80M': {
            'GC3': 0.9395, 'GC3_std': 0.0088,
            'CAI': 0.482, 'CAI_std': 0.0546,
            'Position': -1.899, 'Position_std': 0.3165,
            'Pathogenicity\n(LR AUC)': 0.6823, 'Pathogenicity_std': 0.0047,
        },
        'CodonBERT-HF': {
            'GC3': 0.7217, 'GC3_std': 0.0282,
            'CAI': 0.0634, 'CAI_std': 0.1225,
            'Position': -0.9732, 'Position_std': 0.1346,
            'Pathogenicity\n(LR AUC)': 0.6593, 'Pathogenicity_std': 0.0065,
        },
    }

    models = ['CodonBERT', 'EnCodon-80M', 'CodonBERT-HF']
    model_labels = ['CodonBERT', 'EnCodon\n80M', 'CodonBERT\n-HF']
    attributes = ['GC3', 'CAI', 'Position', 'Pathogenicity\n(LR AUC)']
    attr_keys = ['GC3', 'CAI', 'Position', 'Pathogenicity\n(LR AUC)']

    # Color scheme: GC3=strong blue, CAI=medium blue, Position=gray, Pathogenicity=orange
    attr_colors = ['#2166AC', '#4393C3', '#999999', '#E08214']

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [2, 1]})

    # Left panel: R² for GC3, CAI, Position (all on same scale)
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.22
    r2_attrs = ['GC3', 'CAI', 'Position']
    r2_colors = ['#2166AC', '#4393C3', '#999999']
    r2_labels = ['GC3 (R²)', 'CAI (R²)', 'Position (R²)']

    for j, (attr, color, label) in enumerate(zip(r2_attrs, r2_colors, r2_labels)):
        vals = [data[m][attr] for m in models]
        stds = [data[m][f'{attr}_std'] for m in models]
        offset = (j - 1) * width
        bars = ax1.bar(x + offset, vals, width, yerr=stds,
                       label=label, color=color,
                       edgecolor=BLACK, linewidth=0.6, capsize=2, error_kw={'linewidth': 0.6})

        for i, v in enumerate(vals):
            y_pos = v + stds[i] + 0.03 if v >= 0 else v - stds[i] - 0.08
            ax1.text(x[i] + offset, y_pos, f'{v:.2f}',
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=7, color=color, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=9)
    ax1.set_ylabel('R²', fontsize=10)
    ax1.set_title('Embedding → Attribute Regression', fontsize=11, fontweight='bold', pad=10)
    ax1.axhline(y=0, color=BLACK, linewidth=0.8)
    ax1.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.6, alpha=0.4)
    ax1.set_ylim(-2.5, 1.15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=9)
    ax1.legend(fontsize=8, loc='lower left', framealpha=0.9)

    # Right panel: LR AUC for pathogenicity (zoomed in)
    ax2 = axes[1]
    patho_vals = [data[m]['Pathogenicity\n(LR AUC)'] for m in models]
    patho_stds = [data[m]['Pathogenicity_std'] for m in models]

    bars = ax2.bar(x, patho_vals, 0.5, yerr=patho_stds,
                   color=ORANGE, edgecolor=BLACK, linewidth=0.8,
                   capsize=3, error_kw={'linewidth': 0.8})

    for i, v in enumerate(patho_vals):
        ax2.text(x[i], v + patho_stds[i] + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=ORANGE)

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=9)
    ax2.set_ylabel('LR AUC', fontsize=10)
    ax2.set_title('Pathogenicity\n(Linear Probe)', fontsize=11, fontweight='bold', pad=10)
    ax2.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0, 1.0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=9)

    # Add "near chance" annotation
    ax2.annotate('chance', xy=(1.6, 0.505), fontsize=7, color=GRAY, style='italic')

    fig.tight_layout(w_pad=2)

    for ext in ['pdf', 'png']:
        outpath = os.path.join(FIG_DIR, f'fig9_interpretability.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f'Saved: {outpath}')
    plt.close(fig)


if __name__ == '__main__':
    print('Generating Fig 7: Context ablation...')
    fig7_context_ablation()
    print()
    print('Generating Fig 9: GC3/CAI interpretability...')
    fig9_interpretability()
    print()
    print('Done!')