"""V26 compact forest plot: Task 1 (MisPath) left, Task 2 (SynPath) right.
Uses family_colors and family_markers from gen_fig1_fig2_nature.py.
Models ordered by tokenization type (amino-acid -> BPE -> char -> codon, top to bottom).
Within each type, sorted by SynPath AUC descending.
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, 'results')
FIG_DIR = os.path.join(BASE, 'paper', 'figures_v2', 'v26')
os.makedirs(FIG_DIR, exist_ok=True)

WHITE = '#FFFFFF'
BLACK = '#2D2D2D'

family_colors = {
    'codon-BERT': '#4393C3',
    'codon-EnCodon': '#92C5DE',
    'aa-codon': '#5AAE61',
    'char-codon': '#A6D854',
    'moe-codon': '#FFD92F',
    'DNA-BPE': '#F4A582',
    'traditional': '#999999',
    'protein': '#D6604D',
    'contrastive': '#7B9ED4',
}

family_markers = {
    'codon-BERT': 'o',
    'codon-EnCodon': 's',
    'aa-codon': 'D',
    'char-codon': 'p',
    'moe-codon': 'h',
    'DNA-BPE': '^',
    'traditional': 'v',
    'protein': 'P',
    'contrastive': 'X',
}

MODELS = [
    ('EnCodon-620M',     'codon-EnCodon', 'cLM (codon)'),
    ('CodonBERT',        'codon-BERT',    'cLM (codon)'),
    ('CodonTransformer', 'aa-codon',      'cLM (aa-codon)'),
    ('CodonBERT-HF',     'codon-BERT',    'cLM (codon-RNA)'),
    ('EnCodon-80M',      'codon-EnCodon', 'cLM (codon)'),
    ('CaLM',             'contrastive',   'cLM (contrastive)'),
    ('Mistral-117M',     'moe-codon',     'cLM (MoE)'),
    ('Mistral-16M',      'moe-codon',     'cLM (MoE)'),
    ('Mistral-1M',       'moe-codon',     'cLM (MoE)'),
    ('cdsBERT',          'char-codon',    'CDS LM (char)'),
    ('cdsBERT-plus',     'char-codon',    'CDS LM (char)'),
    ('NT-v2-500M',       'DNA-BPE',       'DNA LM'),
    ('NT-v2-50M',        'DNA-BPE',       'DNA LM'),
    ('ESM-2-650M',       'protein',       'pLM'),
    ('ESM-1b',           'protein',       'pLM'),
    ('kmer6',            'traditional',   'Traditional'),
    ('combined',         'traditional',   'Traditional'),
    ('onehot_freq',      'traditional',   'Traditional'),
    ('kmer4',            'traditional',   'Traditional'),
    ('kmer3',            'traditional',   'Traditional'),
]

DATA = {
    'EnCodon-620M':      {'mis': 0.617, 'syn': 0.785},
    'CodonBERT':          {'mis': 0.660, 'syn': 0.734},
    'CodonTransformer':   {'mis': 0.694, 'syn': 0.713},
    'CodonBERT-HF':       {'mis': 0.659, 'syn': 0.705},
    'EnCodon-80M':        {'mis': 0.633, 'syn': 0.686},
    'CaLM':               {'mis': 0.689, 'syn': 0.673},
    'Mistral-117M':       {'mis': 0.611, 'syn': 0.656},
    'cdsBERT-plus':       {'mis': 0.623, 'syn': 0.601},
    'cdsBERT':            {'mis': 0.629, 'syn': 0.598},
    'Mistral-16M':        {'mis': 0.671, 'syn': 0.598},
    'Mistral-1M':         {'mis': 0.666, 'syn': 0.591},
    'ESM-2-650M':         {'mis': 0.719, 'syn': 0.680},
    'ESM-1b':             {'mis': 0.711, 'syn': 0.600},
    'NT-v2-500M':         {'mis': 0.570, 'syn': 0.699},
    'NT-v2-50M':          {'mis': 0.573, 'syn': 0.643},
    'onehot_pos':         {'mis': 0.755, 'syn': 0.891},
    'kmer6':              {'mis': 0.651, 'syn': 0.700},
    'onehot_freq':        {'mis': 0.667, 'syn': 0.589},
    'combined':           {'mis': 0.666, 'syn': 0.594},
    'kmer4':              {'mis': 0.627, 'syn': 0.580},
    'kmer3':              {'mis': 0.621, 'syn': 0.562},
}

SECTION_BOUNDARIES = {
    'codon': (0, 5),
    'contrastive': (5, 6),
    'moe': (6, 9),
    'char': (9, 11),
    'DNA-BPE': (11, 13),
    'protein': (13, 15),
    'traditional': (15, 20),
}


def load_bootstrap():
    p = os.path.join(RESULTS_DIR, 'bootstrap_ci_results.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def get_ci(bootstrap, model, task):
    if model in bootstrap and task in bootstrap[model]:
        d = bootstrap[model][task]
        if 'ci_lower' in d and 'ci_upper' in d:
            return d['ci_lower'], d['ci_upper']
    return None


def make_compact_forest():
    bootstrap = load_bootstrap()

    n_models = len(MODELS)
    n_total = n_models + 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5.0), sharey=True)

    for ax, task_key, ref_auc, title in [
        (ax1, 'mis', 0.719, 'Task 1: Missense (P2)'),
        (ax2, 'syn', 0.680, 'Task 2: Synonymous (P1)'),
    ]:
        for i, (name, family, label) in enumerate(MODELS):
            if name not in DATA:
                continue
            auc_val = DATA[name][task_key]
            delta = auc_val - ref_auc
            y = n_total - i - 1
            color = family_colors.get(family, '#888888')
            marker = family_markers.get(family, 'o')

            ci = get_ci(bootstrap, name, task_key)
            if ci is not None:
                err_lo = max(delta - ci[0], 0)
                err_hi = max(ci[1] - delta, 0)
                ax.errorbar(delta, y, xerr=[[err_lo], [err_hi]],
                           fmt=marker, color=color, markersize=4,
                           capsize=1.5, capthick=0.7, elinewidth=0.7,
                           markeredgecolor='black', markeredgewidth=0.25, zorder=3)
            else:
                ax.scatter(delta, y, marker=marker, color=color, s=25,
                          edgecolors='black', linewidths=0.25, zorder=3)

        if 'onehot_pos' in DATA:
            delta_op = DATA['onehot_pos'][task_key] - ref_auc
            ax.scatter(delta_op, 0, marker='D', color='#999999', s=30,
                      edgecolors='black', linewidths=0.4, zorder=3)

        ax.axvline(0, color=BLACK, linewidth=1, linestyle='--', zorder=1, alpha=0.6)
        ax.axvspan(-0.2, 0, alpha=0.08, color='#D6604D', zorder=0)
        ax.axvspan(0, 0.2, alpha=0.08, color='#4393C3', zorder=0)
        ax.set_xlabel('\u0394AUC vs ESM-2', fontsize=8, fontweight='bold')
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)

        ax.grid(axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=5.5)
        ax.set_xlim(-0.20, 0.22)

    y_labels = []
    y_positions = []
    for i, (name, family, label) in enumerate(MODELS):
        y = n_total - i - 1
        short = name.replace('-650M', '').replace('-v2-500M', ' (500M)').replace('-v2-50M', ' (50M)') \
                    .replace('-620M', ' (620M)').replace('-80M', ' (80M)').replace('-117M', ' (117M)') \
                    .replace('-16M', ' (16M)').replace('-1M', ' (1M)')
        y_labels.append(short)
        y_positions.append(y)
    y_labels.append('onehot_pos \u2020')
    y_positions.append(0)

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(y_labels, fontsize=5)

    sep_positions = []
    prev_family = None
    for i, (name, family, label) in enumerate(MODELS):
        if prev_family is not None and family != prev_family:
            y_above = n_total - i
            sep_positions.append(y_above)
        prev_family = family

    for y_sep in sep_positions:
        ax1.axhline(y=y_sep - 0.5, color=BLACK, linewidth=0.4, linestyle=':', alpha=0.6)
        ax2.axhline(y=y_sep - 0.5, color=BLACK, linewidth=0.4, linestyle=':', alpha=0.6)

    ax1.axhline(y=0.5, color=BLACK, linewidth=0.5, linestyle=':', alpha=0.6)
    ax2.axhline(y=0.5, color=BLACK, linewidth=0.5, linestyle=':', alpha=0.6)

    legend_items = [
        mpatches.Patch(facecolor=family_colors['codon-BERT'], edgecolor='black', label='cLM (codon-BERT)'),
        mpatches.Patch(facecolor=family_colors['codon-EnCodon'], edgecolor='black', label='cLM (codon-EnCodon)'),
        mpatches.Patch(facecolor=family_colors['aa-codon'], edgecolor='black', label='cLM (aa-codon)'),
        mpatches.Patch(facecolor=family_colors['contrastive'], edgecolor='black', label='cLM (contrastive)'),
        mpatches.Patch(facecolor=family_colors['moe-codon'], edgecolor='black', label='cLM (MoE)'),
        mpatches.Patch(facecolor=family_colors['char-codon'], edgecolor='black', label='CDS LM (char)'),
        mpatches.Patch(facecolor=family_colors['protein'], edgecolor='black', label='pLM'),
        mpatches.Patch(facecolor=family_colors['DNA-BPE'], edgecolor='black', label='DNA LM (BPE)'),
    ]
    fig.legend(handles=legend_items, loc='center', ncol=4, fontsize=5.5,
               frameon=True, fancybox=True, bbox_to_anchor=(0.52, 0.76),
               edgecolor='#AAAAAA', handlelength=1.2, handleheight=0.8,
               facecolor='white', framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 1])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig2_forest_compact.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"Saved to {FIG_DIR}/fig2_forest_compact.png")


if __name__ == '__main__':
    make_compact_forest()
