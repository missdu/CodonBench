"""
CodonBench v12: Unified figure generation script.
Generates all figures from JSON data files (21 models).
Nature MI style: clean, high DPI, consistent color scheme.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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

C_CLM = '#3B7DD8'
C_CLM2 = '#6BA3E6'
C_CLM3 = '#9DC4F0'
C_PLM = '#E64B35'
C_DNA = '#F39B30'
C_TRAD = '#666666'
C_TRAD2 = '#999999'
C_BENIGN = '#00A087'
C_PATHO = '#E64B35'
C_LORA = '#7B2D8E'

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures_v2')
os.makedirs(FIG_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

family_colors = {
    'codon-BERT': '#4393C3',
    'codon-EnCodon': '#92C5DE',
    'aa-codon': '#5AAE61',
    'char-codon': '#A6D854',
    'moe-codon': '#FFD92F',
    'DNA-BPE': '#F4A582',
    'traditional': '#999999',
    'protein': '#D6604D',
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
}


def load_json(fname):
    fpath = os.path.join(RESULTS_DIR, fname)
    for enc in ['utf-8-sig', 'utf-16', 'utf-16-le', 'utf-8', 'latin-1']:
        try:
            with open(fpath, 'r', encoding=enc) as f:
                return json.load(f)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot decode {fname}")


def load_all_data():
    comp = load_json('comprehensive_comparison.json')
    bootstrap = load_json('bootstrap_ci_results.json')
    esm2_t2 = load_json('esm2_task2.json')
    esm2_t3 = load_json('esm2_task3.json')
    esm1b = load_json('esm1b_results.json')
    lora = load_json('lora_finetune_results.json')
    lora_ext = load_json('lora_extended_results.json')
    encodon620m_t2 = load_json('encodon-620m_task2_missense.json')
    encodon620m_t3 = load_json('encodon-620m_task3_synonymous.json')
    codontransformer = load_json('codontransformer_summary.json')
    calm = load_json('calm_summary.json')
    cdsbert = load_json('cdsbert_char_summary.json')
    mistral = load_json('mistral_codon_summary.json')
    cka_reval = load_json('cka_revalidation.json')
    cka_ext = load_json('cka_extended_results.json')
    return {
        'comp': comp, 'bootstrap': bootstrap,
        'esm2_t2': esm2_t2, 'esm2_t3': esm2_t3, 'esm1b': esm1b,
        'lora': lora, 'lora_ext': lora_ext,
        'encodon620m_t2': encodon620m_t2, 'encodon620m_t3': encodon620m_t3,
        'codontransformer': codontransformer, 'calm': calm,
        'cdsbert': cdsbert, 'mistral': mistral,
        'cka_reval': cka_reval, 'cka_ext': cka_ext,
    }


def build_21_models(data):
    comp = data['comp']
    esm2_t2 = data['esm2_t2']
    esm2_t3 = data['esm2_t3']
    esm1b = data['esm1b']
    encodon620m_t2 = data['encodon620m_t2']
    encodon620m_t3 = data['encodon620m_t3']
    codontransformer = data['codontransformer']
    calm = data['calm']
    cdsbert = data['cdsbert']
    mistral = data['mistral']

    models = {}
    for m in comp:
        name = m['model']
        task = m['task']
        auc = m['auc_mean']
        std = m.get('auc_std', 0)
        mtype = m['type']
        if name not in models:
            models[name] = {'t1': None, 't2': None, 't1_std': 0, 't2_std': 0, 'type': mtype}
        if task == 'task2_missense':
            models[name]['t1'] = auc
            models[name]['t1_std'] = std
        elif task == 'task3_synonymous':
            models[name]['t2'] = auc
            models[name]['t2_std'] = std

    display_map = {
        'encodon-80m': ('EnCodon-80M', 'codon-EnCodon'),
        'codonbert': ('CodonBERT', 'codon-BERT'),
        'codonbert_hf': ('CodonBERT-HF', 'codon-BERT'),
        'nt-50m': ('NT-50M', 'DNA-BPE'),
        'nt-500m': ('NT-500M', 'DNA-BPE'),
        'onehot_freq': ('OneHot-freq', 'traditional'),
        'onehot_pos': ('OneHot-pos', 'traditional'),
        'kmer3': ('3-mer', 'traditional'),
        'kmer4': ('4-mer', 'traditional'),
        'kmer6': ('6-mer', 'traditional'),
        'combined': ('Combined', 'traditional'),
    }

    result = {}
    for internal, (display, family) in display_map.items():
        if internal in models:
            result[display] = {
                't1': models[internal]['t1'],
                't2': models[internal]['t2'],
                't1_std': models[internal]['t1_std'],
                't2_std': models[internal]['t2_std'],
                'type': models[internal]['type'],
                'family': family,
            }

    result['EnCodon-620M'] = {
        't1': encodon620m_t2['lr_auc_mean'], 't2': encodon620m_t3['lr_auc_mean'],
        't1_std': encodon620m_t2.get('lr_auc_std', 0), 't2_std': encodon620m_t3.get('lr_auc_std', 0),
        'type': 'cLM', 'family': 'codon-EnCodon',
    }

    ct_t2 = [x for x in codontransformer if x['task'] == 'task2_missense'][0]
    ct_t3 = [x for x in codontransformer if x['task'] == 'task3_synonymous'][0]
    result['CodonTransformer'] = {
        't1': ct_t2['lr_auc_mean'], 't2': ct_t3['lr_auc_mean'],
        't1_std': ct_t2.get('lr_auc_std', 0), 't2_std': ct_t3.get('lr_auc_std', 0),
        'type': 'cLM', 'family': 'aa-codon',
    }

    calm_t2 = [x for x in calm if x['task'] == 'task2_missense'][0]
    calm_t3 = [x for x in calm if x['task'] == 'task3_synonymous'][0]
    result['CaLM'] = {
        't1': calm_t2['lr_auc_mean'], 't2': calm_t3['lr_auc_mean'],
        't1_std': calm_t2.get('lr_auc_std', 0), 't2_std': calm_t3.get('lr_auc_std', 0),
        'type': 'cLM', 'family': 'codon-BERT',
    }

    for entry in cdsbert:
        mname = entry['model']
        display = mname
        task = entry['task']
        if display not in result:
            result[display] = {
                't1': None, 't2': None, 't1_std': 0, 't2_std': 0,
                'type': 'cLM (char)', 'family': 'char-codon',
            }
        if task == 'task2_missense':
            result[display]['t1'] = entry['lr_auc_mean']
            result[display]['t1_std'] = entry.get('lr_auc_std', 0)
        elif task == 'task3_synonymous':
            result[display]['t2'] = entry['lr_auc_mean']
            result[display]['t2_std'] = entry.get('lr_auc_std', 0)

    mistral_display_map = {
        'mistral-codon-117m': ('Mistral-117M', 'moe-codon'),
        'mistral-codon-16m': ('Mistral-16M', 'moe-codon'),
        'mistral-codon-1m': ('Mistral-1M', 'moe-codon'),
    }
    for entry in mistral:
        internal = entry['model']
        if internal not in mistral_display_map:
            continue
        display, family = mistral_display_map[internal]
        task = entry['task']
        if display not in result:
            result[display] = {
                't1': None, 't2': None, 't1_std': 0, 't2_std': 0,
                'type': 'cLM (MoE)', 'family': family,
            }
        if task == 'task2_missense':
            result[display]['t1'] = entry['lr_auc_mean']
            result[display]['t1_std'] = entry.get('lr_auc_std', 0)
        elif task == 'task3_synonymous':
            result[display]['t2'] = entry['lr_auc_mean']
            result[display]['t2_std'] = entry.get('lr_auc_std', 0)

    ESM2_T1 = esm2_t2['ROC-AUC_mean']
    ESM2_T2 = esm2_t3['ROC-AUC_mean']
    ESM1B_T1 = esm1b['task2_missense']['ROC-AUC_mean']
    ESM1B_T2 = esm1b['task3_synonymous']['ROC-AUC_mean']
    result['ESM-2'] = {
        't1': ESM2_T1, 't2': ESM2_T2,
        't1_std': esm2_t2.get('ROC-AUC_std', 0), 't2_std': esm2_t3.get('ROC-AUC_std', 0),
        'type': 'pLM', 'family': 'protein',
    }
    result['ESM-1b'] = {
        't1': ESM1B_T1, 't2': ESM1B_T2,
        't1_std': 0, 't2_std': 0,
        'type': 'pLM', 'family': 'protein',
    }

    return result, ESM2_T1, ESM2_T2, ESM1B_T1, ESM1B_T2


def build_lora_data(data):
    lora = data['lora']
    lora_ext = data['lora_ext']

    result = {}
    for entry in lora:
        m = entry['model']
        task = entry['task']
        if m not in result:
            result[m] = {}
        if task == 'task2_missense':
            result[m]['t1_lr'] = entry['test_lr_auc']
            result[m]['t1_mlp'] = entry['test_mlp_auc']
            result[m]['t1_lora'] = entry['test_lora_auc']
        elif task == 'task3_synonymous':
            result[m]['t2_lr'] = entry['test_lr_auc']
            result[m]['t2_mlp'] = entry['test_mlp_auc']
            result[m]['t2_lora'] = entry['test_lora_auc']

    for entry in lora_ext:
        m = entry['model']
        if not entry.get('success', False):
            continue
        task = entry['task']
        if m not in result:
            result[m] = {}
        if task == 'task2_missense':
            result[m]['t1_lora'] = entry['test_auc']
        elif task == 'task3_synonymous':
            result[m]['t2_lora'] = entry['test_auc']

    enc620m_t2 = data['encodon620m_t2']
    enc620m_t3 = data['encodon620m_t3']
    if 'encodon-620m' not in result:
        result['encodon-620m'] = {}
    result['encodon-620m']['t1_lr'] = enc620m_t2.get('lr_auc_mean', None)
    result['encodon-620m']['t1_mlp'] = enc620m_t2.get('mlp_auc_test', None)
    result['encodon-620m']['t2_lr'] = enc620m_t3.get('lr_auc_mean', None)
    result['encodon-620m']['t2_mlp'] = enc620m_t3.get('mlp_auc_test', None)

    ct = data['codontransformer']
    if 'codontransformer' not in result:
        result['codontransformer'] = {}
    for entry in ct:
        task = entry['task']
        if task == 'task2_missense':
            result['codontransformer']['t1_lr'] = entry.get('lr_auc_mean', None)
            result['codontransformer']['t1_mlp'] = entry.get('mlp_auc_test', None)
        elif task == 'task3_synonymous':
            result['codontransformer']['t2_lr'] = entry.get('lr_auc_mean', None)
            result['codontransformer']['t2_mlp'] = entry.get('mlp_auc_test', None)

    calm = data['calm']
    if 'calm' not in result:
        result['calm'] = {}
    for entry in calm:
        task = entry['task']
        if task == 'task2_missense':
            result['calm']['t1_lr'] = entry.get('lr_auc_mean', None)
            result['calm']['t1_mlp'] = entry.get('mlp_auc_test', None)
        elif task == 'task3_synonymous':
            result['calm']['t2_lr'] = entry.get('lr_auc_mean', None)
            result['calm']['t2_mlp'] = entry.get('mlp_auc_test', None)

    return result


def build_cka_matrix(data):
    cka_reval = data['cka_reval']
    cka_ext = data['cka_ext']

    model_order = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M', 'EnCodon-620M', 'CodonTransformer', 'CaLM']
    n = len(model_order)
    lin_mat = np.eye(n)
    rbf_mat = np.eye(n)

    internal_map = {
        'codonbert': 'CodonBERT', 'codonbert_hf': 'CodonBERT-HF',
        'encodon-80m': 'EnCodon-80M', 'encodon-620m': 'EnCodon-620M',
        'codontransformer': 'CodonTransformer', 'calm': 'CaLM',
    }

    for entry in cka_reval:
        m1 = internal_map.get(entry['model1'], entry['model1'])
        m2 = internal_map.get(entry['model2'], entry['model2'])
        if m1 in model_order and m2 in model_order:
            i, j = model_order.index(m1), model_order.index(m2)
            lin_mat[i, j] = lin_mat[j, i] = entry['linear_cka']
            rbf_mat[i, j] = rbf_mat[j, i] = entry['rbf_cka']

    for entry in cka_ext:
        m1 = internal_map.get(entry['model1'], entry['model1'])
        m2 = internal_map.get(entry['model2'], entry['model2'])
        if m1 in model_order and m2 in model_order:
            i, j = model_order.index(m1), model_order.index(m2)
            lin_mat[i, j] = lin_mat[j, i] = entry['linear_cka']
            rbf_mat[i, j] = rbf_mat[j, i] = entry['rbf_cka']

    return model_order, lin_mat, rbf_mat


def get_bootstrap_ci(bootstrap, model_name, task, reference='ESM-2'):
    key_map = {
        'EnCodon-620M': 'encodon-620m', 'CodonBERT': 'codonbert', 'CodonBERT-HF': 'codonbert_hf',
        'EnCodon-80M': 'encodon-80m', 'CodonTransformer': 'codontransformer', 'CaLM': 'calm',
        'NT-50M': 'nt-50m', 'NT-500M': 'nt-500m', 'OneHot-pos': 'onehot_pos',
        'OneHot-freq': 'onehot_freq', '6-mer': 'kmer6', 'Combined': 'combined',
        'ESM-2': 'esm2', 'ESM-1b': 'esm1b',
        'cdsBERT': 'cdsbert', 'cdsBERT-plus': 'cdsbert_plus',
        'Mistral-117M': 'mistral-117m', 'Mistral-16M': 'mistral-16m', 'Mistral-1M': 'mistral-1m',
        '3-mer': 'kmer3', '4-mer': 'kmer4',
    }
    internal = key_map.get(model_name, model_name.lower())
    for b in bootstrap:
        if b['task'] != task:
            continue
        parts = b['comparison'].split(' vs ')
        left = parts[0].strip().lower()
        right = parts[1].strip().lower()
        ref_lower = reference.lower()
        if left == internal.lower() and right == ref_lower:
            return (b['delta_auc'], b['ci_95_lo'], b['ci_95_hi'])
        if right == internal.lower() and left == ref_lower:
            return (-b['delta_auc'], -b['ci_95_hi'], -b['ci_95_lo'])
    return None


# ═══════════════════════════════════════════════════════════════════
# FIG 1: Overview schematic (4-panel)
# ═══════════════════════════════════════════════════════════════════

def make_fig1():
    fig = plt.figure(figsize=(18, 5.5))

    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.2, 1.0, 1.4], wspace=0.25,
                          left=0.03, right=0.97, top=0.88, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    ax_a.set_xlim(-3.5, 3.5)
    ax_a.set_ylim(-2.5, 4.5)
    ax_a.axis('off')
    ax_a.set_title('a  Information decomposition', fontsize=10, fontweight='bold', pad=8, loc='left')

    outer = mpatches.Ellipse((0, 1.2), 6.2, 5.8, facecolor='#F5F5F5', edgecolor=BLACK, linewidth=1.8, alpha=0.35)
    ax_a.add_patch(outer)
    ax_a.text(0, 3.8, r'$I(\mathrm{CDS}; Y)$', fontsize=13, ha='center', va='center', fontweight='bold')

    aa_ellipse = mpatches.Ellipse((-1.0, 0.8), 3.4, 3.8, facecolor=BLUE, edgecolor=BLUE, linewidth=1.8, alpha=0.25)
    ax_a.add_patch(aa_ellipse)
    ax_a.text(-1.0, 1.6, r'$I(A; Y)$', fontsize=11, ha='center', va='center', color=BLUE, fontweight='bold')
    ax_a.text(-1.0, 0.7, 'amino-acid\nchannel', fontsize=7, ha='center', va='center', color=BLUE, style='italic')

    syn_ellipse = mpatches.Ellipse((1.0, 0.8), 3.4, 3.8, facecolor=ORANGE, edgecolor=ORANGE, linewidth=1.8, alpha=0.25)
    ax_a.add_patch(syn_ellipse)
    ax_a.text(1.0, 1.6, r'$I(\sigma; Y|A)$', fontsize=11, ha='center', va='center', color='#B35806', fontweight='bold')
    ax_a.text(1.0, 0.7, 'synonymous-choice\nchannel', fontsize=7, ha='center', va='center', color='#B35806', style='italic')

    ax_a.text(0, -0.3, 'overlap\n(variant position)', fontsize=6.5, ha='center', va='center', color=GRAY, style='italic')
    ax_a.annotate('', xy=(0, -1.5), xytext=(0, -2.1), arrowprops=dict(arrowstyle='->', color=BLACK, lw=1.5))
    ax_a.text(0, -2.3, 'CDS input', fontsize=8, ha='center', va='center', fontweight='bold')

    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 8)
    ax_b.axis('off')
    ax_b.set_title('b  Task-channel mapping', fontsize=10, fontweight='bold', pad=8, loc='left')

    tasks = [
        ('Task 1', 'Missense\npathogenicity', 'Amino-acid', 'P2'),
        ('Task 2', 'Synonymous\npathogenicity', 'Synonymous-\nchoice', 'P1'),
        ('Task 3', 'mRFP expression\n(within-protein)', 'Both', 'P1+P2'),
        ('Task 4', 'E. coli expression\n(cross-protein)', 'I(A;Y)\ndominant', 'P2'),
        ('Task 5', 'mRNA stability', 'Both', 'P1+P2'),
        ('Task 6', 'Fungal expression\n(cross-protein)', 'I(\u03c3;Y|A)\nsubstantial', 'P1'),
    ]

    header_y = 7.2
    col_x = [0.2, 1.5, 4.5, 7.0, 9.0]
    row_h = 0.95

    ax_b.text(col_x[0], header_y, 'Task', fontsize=7.5, fontweight='bold', va='center')
    ax_b.text(col_x[1], header_y, 'Description', fontsize=7.5, fontweight='bold', va='center')
    ax_b.text(col_x[2], header_y, 'Channel', fontsize=7.5, fontweight='bold', va='center')
    ax_b.text(col_x[3], header_y, 'Prediction', fontsize=7.5, fontweight='bold', va='center')
    ax_b.plot([0.1, 9.8], [header_y - 0.3, header_y - 0.3], color=BLACK, linewidth=1)

    for i, (task_id, desc, ch_label, pred) in enumerate(tasks):
        y = header_y - 0.8 - i * row_h
        if ch_label == 'Amino-acid' or 'I(A;Y)' in ch_label:
            ch_color = BLUE
        elif 'Synonymous' in ch_label or 'I(\u03c3;Y|A)' in ch_label:
            ch_color = ORANGE
        else:
            ch_color = '#7B2D8E'
        if pred == 'P1':
            badge_color = ORANGE
        elif pred == 'P2':
            badge_color = BLUE
        else:
            badge_color = '#7B2D8E'
        ax_b.text(col_x[0], y, task_id, fontsize=7, va='center', fontweight='bold')
        ax_b.text(col_x[1], y, desc, fontsize=6.5, va='center')
        ax_b.text(col_x[2], y, ch_label, fontsize=6.5, va='center', color=ch_color, fontweight='bold')
        ax_b.text(col_x[3] + 0.4, y, pred, fontsize=7, ha='center', va='center',
                  color=WHITE, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor=badge_color, edgecolor=badge_color))
        if i < len(tasks) - 1:
            ax_b.plot([0.1, 9.8], [y - 0.4, y - 0.4], color='#E0E0E0', linewidth=0.5)

    ax_b.text(5, 0.3, 'P1: cLMs excel  |  P2: pLMs excel', fontsize=6.5, ha='center', va='center',
              fontweight='bold', style='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7F7F7', edgecolor=GRAY, linewidth=1))

    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 8)
    ax_c.axis('off')
    ax_c.set_title('c  Evaluation protocol', fontsize=10, fontweight='bold', pad=8, loc='left')

    levels = [
        ('Level 0', 'Zero-shot LLR', '#E8E8E8', '#666666'),
        ('Level 1', 'Linear probing', LIGHT_BLUE, BLUE),
        ('Level 2', 'Nonlinear probing', LIGHT_ORANGE, '#B35806'),
        ('Level 3', 'LoRA fine-tuning', '#E8D5F5', '#7B2D8E'),
    ]

    for i, (level, desc, bg_color, text_color) in enumerate(levels):
        y = 1.0 + i * 1.6
        x_center = 4.0
        width = 2.8 + i * 0.9
        x_start = x_center - width / 2
        box = mpatches.FancyBboxPatch((x_start, y - 0.4), width, 0.8, boxstyle="round,pad=0.12",
                                       facecolor=bg_color, edgecolor=text_color, linewidth=1.3, alpha=0.85)
        ax_c.add_patch(box)
        ax_c.text(x_center, y + 0.12, level, fontsize=7, ha='center', va='center', color=text_color, fontweight='bold')
        ax_c.text(x_center, y - 0.15, desc, fontsize=6.5, ha='center', va='center', color=text_color)
        if i < 3:
            y_top = y + 0.4
            y_bot = y + 1.6 - 0.4
            ax_c.annotate('', xy=(x_center, y_bot), xytext=(x_center, y_top),
                          arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.0))

    ax_c.annotate('', xy=(8.5, 7.0), xytext=(8.5, 1.0),
                  arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.3))
    ax_c.text(8.8, 4.0, 'Increasing\ninformation\naccessibility', fontsize=6, ha='left', va='center',
              color=GRAY, fontweight='bold', rotation=90)

    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 8)
    ax_d.axis('off')
    ax_d.set_title('d  Tokenization determines extractable depth', fontsize=10, fontweight='bold', pad=8, loc='left')

    model_rows = [
        ('Codon-level\ncLM', 'CodonBERT, EnCodon', True, True),
        ('Character-level\ncLM', 'cdsBERT, CodonBERT-v2', True, False),
        ('MoE cLM', 'Mistral-Codon', True, False),
        ('AA-codon\ncLM', 'CodonTransformer', True, True),
        ('DNA LM\n(BPE)', 'Nucleotide Transformer', True, False),
        ('Protein LM', 'ESM-2, ESM-1b', True, False),
        ('Traditional', 'OneHot, k-mer', True, False),
    ]

    header_y_d = 7.0
    col_x_d = [0.2, 2.2, 5.5, 7.8]

    ax_d.text(col_x_d[0], header_y_d, 'Category', fontsize=7, fontweight='bold', va='center')
    ax_d.text(col_x_d[1], header_y_d, 'Examples', fontsize=7, fontweight='bold', va='center')
    ax_d.text(col_x_d[2], header_y_d, 'Amino-acid\nchannel', fontsize=6.5, fontweight='bold', va='center', ha='center')
    ax_d.text(col_x_d[3], header_y_d, 'Synonymous-\nchoice channel', fontsize=6.5, fontweight='bold', va='center', ha='center')
    ax_d.plot([0.1, 9.8], [header_y_d - 0.45, header_y_d - 0.45], color=BLACK, linewidth=1)

    for i, (cat, examples, has_aa, has_syn) in enumerate(model_rows):
        y = header_y_d - 1.0 - i * 0.85
        ax_d.text(col_x_d[0], y, cat, fontsize=6.5, va='center', fontweight='bold')
        ax_d.text(col_x_d[1], y, examples, fontsize=6, va='center', color=GRAY)
        check_fs = 10
        if has_aa:
            ax_d.text(col_x_d[2], y, '\u2713', fontsize=check_fs, ha='center', va='center', color=BLUE, fontweight='bold')
        else:
            ax_d.text(col_x_d[2], y, '\u2717', fontsize=check_fs, ha='center', va='center', color='#CCCCCC', fontweight='bold')
        if has_syn:
            ax_d.text(col_x_d[3], y, '\u2713', fontsize=check_fs, ha='center', va='center', color=ORANGE, fontweight='bold')
        else:
            ax_d.text(col_x_d[3], y, '\u2717', fontsize=check_fs, ha='center', va='center', color='#CCCCCC', fontweight='bold')
        if i < len(model_rows) - 1:
            ax_d.plot([0.1, 9.8], [y - 0.3, y - 0.3], color='#E0E0E0', linewidth=0.5)

    key_insight_y = header_y_d - len(model_rows) * 0.85 - 0.9
    ax_d.text(5, key_insight_y, 'Codon tokenization encodes I(σ;Y|A)\nat nonlinearly-accessible depth',
              fontsize=7, ha='center', va='center', fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF3E0', edgecolor=ORANGE, linewidth=1.3))

    fig.text(0.235, 0.48, '\u25B6', fontsize=14, ha='center', va='center', color=GRAY)
    fig.text(0.465, 0.48, '\u25B6', fontsize=14, ha='center', va='center', color=GRAY)
    fig.text(0.695, 0.48, '\u25B6', fontsize=14, ha='center', va='center', color=GRAY)

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig1_overview_v2.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig1 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 2: Forest plot (21 models)
# ═══════════════════════════════════════════════════════════════════

def make_fig2(model_auc, bootstrap, ESM2_T1, ESM2_T2):
    model_order = [
        ('EnCodon-620M', 'cLM', 'codon-EnCodon'),
        ('CodonBERT', 'cLM', 'codon-BERT'),
        ('CodonTransformer', 'cLM', 'aa-codon'),
        ('CodonBERT-HF', 'cLM', 'codon-BERT'),
        ('EnCodon-80M', 'cLM', 'codon-EnCodon'),
        ('CaLM', 'cLM', 'codon-BERT'),
        ('cdsBERT', 'cLM (char)', 'char-codon'),
        ('cdsBERT-plus', 'cLM (char)', 'char-codon'),
        ('Mistral-16M', 'cLM (MoE)', 'moe-codon'),
        ('Mistral-117M', 'cLM (MoE)', 'moe-codon'),
        ('Mistral-1M', 'cLM (MoE)', 'moe-codon'),
        ('NT-500M', 'DNA LM', 'DNA-BPE'),
        ('NT-50M', 'DNA LM', 'DNA-BPE'),
        ('OneHot-pos', 'Traditional', 'traditional'),
        ('6-mer', 'Traditional', 'traditional'),
        ('OneHot-freq', 'Traditional', 'traditional'),
        ('Combined', 'Traditional', 'traditional'),
        ('4-mer', 'Traditional', 'traditional'),
        ('3-mer', 'Traditional', 'traditional'),
        ('ESM-2', 'pLM', 'protein'),
        ('ESM-1b', 'pLM', 'protein'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True)

    for ax, task_key, ref_auc, title, panel_label, task_str in [
        (ax1, 't2', ESM2_T2, 'Task 2: Synonymous variant (P1)', 'a', 'task3'),
        (ax2, 't1', ESM2_T1, 'Task 1: Missense variant (P2)', 'b', 'task2'),
    ]:
        for i, (name, mtype, family) in enumerate(model_order):
            if name not in model_auc:
                continue
            auc_val = model_auc[name][task_key]
            if auc_val is None:
                continue

            delta = auc_val - ref_auc
            y = len(model_order) - i - 1
            color = family_colors.get(family, GRAY)
            marker = family_markers.get(family, 'o')

            ci = get_bootstrap_ci(bootstrap, name, task_str, 'ESM-2')

            if ci is not None:
                err_lo = max(delta - ci[1], 0)
                err_hi = max(ci[2] - delta, 0)
                ax.errorbar(delta, y, xerr=[[err_lo], [err_hi]],
                           fmt=marker, color=color, markersize=7,
                           capsize=3, capthick=1.2, elinewidth=1.2,
                           markeredgecolor='black', markeredgewidth=0.5, zorder=3)
            else:
                ax.scatter(delta, y, marker=marker, color=color, s=60,
                          edgecolors='black', linewidths=0.5, zorder=3)

            if ci is not None and ci[1] > 0 and task_key == 't2':
                ax.text(ci[2] + 0.005, y, '*', fontsize=9, color=color, va='center', fontweight='bold')

        ax.axvline(0, color=BLACK, linewidth=1.5, linestyle='--', zorder=1, alpha=0.7)
        ax.axvspan(-0.2, 0, alpha=0.06, color='#D6604D', zorder=0)
        ax.axvspan(0, 0.2, alpha=0.06, color='#4393C3', zorder=0)
        ax.set_xlabel('Delta AUC vs ESM-2', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-0.22, 0.25)
        ax.text(0.02, 1.02, panel_label, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=9)

    ax1.set_yticks(range(len(model_order)))
    ax1.set_yticklabels(list(reversed([n for n, _, _ in model_order])), fontsize=8)

    legend_elements = [
        mpatches.Patch(facecolor=family_colors['codon-BERT'], edgecolor='black', label='codon-BERT family'),
        mpatches.Patch(facecolor=family_colors['codon-EnCodon'], edgecolor='black', label='codon-EnCodon family'),
        mpatches.Patch(facecolor=family_colors['aa-codon'], edgecolor='black', label='aa-codon family'),
        mpatches.Patch(facecolor=family_colors['char-codon'], edgecolor='black', label='char-codon family'),
        mpatches.Patch(facecolor=family_colors['moe-codon'], edgecolor='black', label='MoE-codon family'),
        mpatches.Patch(facecolor=family_colors['DNA-BPE'], edgecolor='black', label='DNA-BPE family'),
        mpatches.Patch(facecolor=family_colors['traditional'], edgecolor='black', label='Traditional'),
        mpatches.Patch(facecolor=family_colors['protein'], edgecolor='black', label='Protein LM'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8, frameon=True, fancybox=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Codon-level models outperform ESM-2 on synonymous variants (21 models)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig2_forest_plot.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig2 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 2b: cLM vs pLM crossover (expanded)
# ═══════════════════════════════════════════════════════════════════

def make_fig2_clm_vs_plm(model_auc):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.8))

    models_show = [
        ('ESM-2', C_PLM, 'pLM'),
        ('ESM-1b', '#F4A582', 'pLM'),
        ('EnCodon-620M', '#92C5DE', 'cLM (codon)'),
        ('CodonBERT', C_CLM, 'cLM (codon)'),
        ('CodonBERT-HF', C_CLM2, 'cLM (codon)'),
        ('EnCodon-80M', C_CLM3, 'cLM (codon)'),
        ('CaLM', '#A6D854', 'cLM (codon)'),
        ('CodonTransformer', '#5AAE61', 'cLM (aa-codon)'),
        ('cdsBERT', '#E6AB02', 'cLM (char)'),
        ('cdsBERT-plus', '#CAB2D6', 'cLM (char)'),
        ('Mistral-117M', '#FFD92F', 'cLM (MoE)'),
        ('Mistral-16M', '#E78AC3', 'cLM (MoE)'),
        ('Mistral-1M', '#B3B3B3', 'cLM (MoE)'),
    ]

    for ax, task_key, title in zip(
        axes, ['t1', 't2'],
        ['Task 1: Missense Variant\nPathogenicity', 'Task 2: Synonymous Variant\nPathogenicity']):

        y_pos = np.arange(len(models_show))
        aucs = []
        colors = []

        for name, color, mtype in models_show:
            if name in model_auc and model_auc[name][task_key] is not None:
                aucs.append(model_auc[name][task_key])
            else:
                aucs.append(0)
            colors.append(color)

        bars = ax.barh(y_pos, aucs, height=0.6, color=colors, edgecolor='white', linewidth=0.3, alpha=0.85)
        ax.axvline(x=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n for n, _, _ in models_show], fontsize=5.5)
        ax.set_xlabel('ROC-AUC')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_xlim(0.45, 0.82)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, auc in enumerate(aucs):
            if auc > 0:
                ax.text(auc + 0.005, i, f'{auc:.3f}', va='center', fontsize=5, fontweight='bold')

    fig.text(0.5, -0.02, 'pLM wins (Task 1)          cLMs win (Task 2)', ha='center', fontsize=5.5,
             fontstyle='italic', color='#666666')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig2_clm_vs_plm.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig2_clm_vs_plm saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 3: Probing Method Ablation
# ═══════════════════════════════════════════════════════════════════

def make_fig3_probing(lora_data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    methods = ['LR (C=0.01)', 'KNN-5', 'LR (C=100)', 'LR (C=1.0)', 'MLP (128->64)']
    aucs = [0.650, 0.679, 0.686, 0.705, 0.867]
    stds = [0.015, 0.009, 0.020, 0.013, 0.012]
    colors_abl = [C_TRAD2, C_TRAD2, C_TRAD2, C_CLM, C_PLM]

    ax = axes[0]
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, aucs, height=0.55, color=colors_abl, edgecolor='white', linewidth=0.3, alpha=0.85,
                    xerr=stds, capsize=2, error_kw={'linewidth': 0.5})
    ax.axvline(x=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=6)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('Probing Method Ablation\n(CodonBERT-HF, Task 2)', fontsize=7, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, auc in enumerate(aucs):
        ax.text(auc + stds[i] + 0.01, i, f'{auc:.3f}', va='center', fontsize=5.5, fontweight='bold')

    ax.annotate('', xy=(0.867, 4.15), xytext=(0.705, 3.85),
                arrowprops=dict(arrowstyle='<->', color=C_PLM, linewidth=1.0))
    ax.text(0.72, 3.6, '+23%', fontsize=6, fontweight='bold', color=C_PLM, ha='center', va='center')

    ax = axes[1]
    models_mlp = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']
    lr_aucs = [0.716, 0.668, 0.687]
    mlp_aucs = [0.849, 0.816, 0.812]
    lora_aucs = [0.912, 0.880, 0.893]

    x_pos = np.arange(len(models_mlp))
    width = 0.25
    ax.bar(x_pos - width, lr_aucs, width, label='LR (Level 1)', color=C_CLM, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos, mlp_aucs, width, label='MLP (Level 2)', color=C_PLM, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + width, lora_aucs, width, label='LoRA (Level 3)', color=C_LORA, alpha=0.7, edgecolor='white', linewidth=0.3)

    for i, (lr, mlp, lora) in enumerate(zip(lr_aucs, mlp_aucs, lora_aucs)):
        ax.text(i - width, lr + 0.01, f'{lr:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_CLM)
        ax.text(i, mlp + 0.01, f'{mlp:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_PLM)
        ax.text(i + width, lora + 0.01, f'{lora:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_LORA)
        pct = (lora - lr) / lr * 100
        ax.text(i, lora + 0.04, f'+{pct:.0f}%\nvs LR', ha='center', fontsize=4.5, fontweight='bold', color=C_LORA)

    ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_mlp, fontsize=6)
    ax.set_ylabel('ROC-AUC (Task 2)')
    ax.set_ylim(0.5, 1.0)
    ax.set_title('LR vs MLP vs LoRA\n(Independent Test Set)', fontsize=7, fontweight='bold')
    ax.legend(loc='lower right', fontsize=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig3_probing_ablation.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig3 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 4: LOGO recoverability dumbbell
# ═══════════════════════════════════════════════════════════════════

def make_fig4(ESM2_T2):
    logo_data = load_json('logo_cv_results.json')

    models_show = ['onehot_pos', 'codonbert', 'codonbert_hf', 'encodon-80m']
    display_names = ['OneHot-pos', 'CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']

    standard_aucs = {'onehot_pos': 0.891, 'codonbert': 0.7338, 'codonbert_hf': 0.6824, 'encodon-80m': 0.6858}
    logo_aucs = {m: logo_data[m]['auc_mean'] for m in models_show}
    lora_aucs = {'onehot_pos': None, 'codonbert': 0.9115, 'codonbert_hf': 0.8796, 'encodon-80m': 0.8929}

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (m, dname) in enumerate(zip(models_show, display_names)):
        y = len(models_show) - i - 1
        std_auc = standard_aucs[m]
        logo_auc = logo_aucs[m]
        lora_auc = lora_aucs[m]

        ax.scatter(std_auc, y, marker='o', color=BLUE, s=100, zorder=3, edgecolors='black', linewidths=0.5, label='Standard LR' if i == 0 else '')
        ax.scatter(logo_auc, y, marker='s', color='#D6604D', s=100, zorder=3, edgecolors='black', linewidths=0.5, label='LOGO LR' if i == 0 else '')
        ax.plot([std_auc, logo_auc], [y, y], color=GRAY, linewidth=1.5, linestyle='-', zorder=2, alpha=0.5)

        drop = std_auc - logo_auc
        mid_x = (std_auc + logo_auc) / 2
        ax.text(mid_x, y + 0.18, f'down{drop:.2f}', fontsize=8, ha='center', va='bottom', color='#D6604D')

        if lora_auc is not None:
            ax.scatter(lora_auc, y + 0.15, marker='D', color='#1A9850', s=80, zorder=3, edgecolors='black', linewidths=0.5, label='LoRA (r=8)' if i == 0 else '')
            ax.plot([logo_auc, lora_auc], [y, y + 0.15], color='#1A9850', linewidth=1, linestyle='--', zorder=2, alpha=0.5)
            recovery = lora_auc - logo_auc
            ax.text(lora_auc + 0.01, y + 0.15, f'+{recovery:.2f}', fontsize=7, ha='left', va='center', color='#1A9850')

    ax.set_yticks(range(len(models_show)))
    ax.set_yticklabels(list(reversed(display_names)), fontsize=11)
    ax.set_xlabel('AUC (Task 2: synonymous variant)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.axvline(ESM2_T2, color='#D6604D', linewidth=1, linestyle=':', alpha=0.5, zorder=1)
    ax.text(ESM2_T2, len(models_show) - 0.3, 'ESM-2', fontsize=8, ha='center', color='#D6604D')
    ax.legend(fontsize=10, loc='lower right', frameon=True, fancybox=True)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=10)
    ax.text(0.02, 1.02, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax.set_title('LOGO-CV reveals position leakage; LoRA recovers cLM performance', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig4_logo_recoverability.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig4 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 5: CKA Similarity Matrix (6x6)
# ═══════════════════════════════════════════════════════════════════

def make_fig5_cka(model_order, lin_mat, rbf_mat):
    n = len(model_order)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, data, title in zip(axes, [lin_mat, rbf_mat], ['Linear CKA', 'RBF CKA']):
        im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(n))
        ax.set_xticklabels(model_order, fontsize=7, rotation=30, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(model_order, fontsize=7)
        ax.set_title(title, fontsize=10, fontweight='bold')

        for i in range(n):
            for j in range(n):
                val = data[i, j]
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, fontweight='bold', color=color)

        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle('Representation Similarity Between cLMs (Task 2: synonymous variant)', fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig5_cka_similarity.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig5 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 6: Oracle ceiling
# ═══════════════════════════════════════════════════════════════════

def make_fig6_oracle(ESM1B_T2, ESM2_T2):
    bars = [
        ('Random', 0.5, GRAY),
        ('ESM-1b (LR)', ESM1B_T2, '#D6604D'),
        ('ESM-2 (LR)', ESM2_T2, '#D6604D'),
        ('Best cLM (LR)\nEnCodon-620M', 0.785, BLUE),
        ('Best cLM (MLP)\nCodonBERT', 0.8486, '#4393C3'),
        ('Best cLM (LoRA)\nCodonTransformer', 0.9425, '#2166AC'),
        ('Oracle (GBT)', 0.949, '#1A9850'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(bars))
    aucs = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    labels = [b[0] for b in bars]

    bars_drawn = ax.barh(y_pos, aucs, color=colors, edgecolor='black', linewidth=0.5, height=0.6, alpha=0.85)
    for i, (auc, bar) in enumerate(zip(aucs, bars_drawn)):
        ax.text(auc + 0.005, i, f'{auc:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(0.9425, 5), xytext=(0.949, 5), arrowprops=dict(arrowstyle='<->', color='#1A9850', lw=1.5))
    ax.text(0.9455, 5.3, '99.4%', fontsize=9, ha='center', color='#1A9850', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('AUC (Task 2: synonymous variant)', fontsize=11, fontweight='bold')
    ax.set_xlim(0.4, 1.02)
    ax.axvline(0.949, color='#1A9850', linewidth=1, linestyle='--', alpha=0.5, zorder=1)
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', labelsize=10)
    ax.text(0.02, 1.02, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax.set_title('cLM with LoRA reaches 99.4% of oracle performance ceiling', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig6_oracle_ceiling.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig6 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 9: LoRA Evaluation Hierarchy (5 cLMs)
# ═══════════════════════════════════════════════════════════════════

def make_fig9_lora(lora_data):
    models_info = [
        ('codonbert', 'CodonBERT'),
        ('codonbert_hf', 'CodonBERT-HF'),
        ('encodon-80m', 'EnCodon-80M'),
        ('encodon-620m', 'EnCodon-620M'),
        ('codontransformer', 'CodonTransformer'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    for ax, lr_key, mlp_key, lora_key, title in zip(
        axes,
        ['t1_lr', 't2_lr'], ['t1_mlp', 't2_mlp'], ['t1_lora', 't2_lora'],
        ['Task 1: Missense\nVariant Pathogenicity', 'Task 2: Synonymous\nVariant Pathogenicity']
    ):
        display_names = []
        lr_vals = []
        mlp_vals = []
        lora_vals = []

        for internal, display in models_info:
            if internal in lora_data:
                d = lora_data[internal]
                lr_v = d.get(lr_key)
                mlp_v = d.get(mlp_key)
                lora_v = d.get(lora_key)
                if lora_v is not None:
                    display_names.append(display)
                    lr_vals.append(lr_v if lr_v else 0)
                    mlp_vals.append(mlp_v if mlp_v else 0)
                    lora_vals.append(lora_v)

        x_pos = np.arange(len(display_names))
        width = 0.22

        bars_lr = ax.bar(x_pos - width, lr_vals, width, label='LR (Level 1)', color=C_CLM, alpha=0.75, edgecolor='white', linewidth=0.3)
        bars_mlp = ax.bar(x_pos, mlp_vals, width, label='MLP (Level 2)', color=C_PLM, alpha=0.75, edgecolor='white', linewidth=0.3)
        bars_lora = ax.bar(x_pos + width, lora_vals, width, label='LoRA (Level 3)', color=C_LORA, alpha=0.75, edgecolor='white', linewidth=0.3)

        for i in range(len(display_names)):
            if lr_vals[i] > 0:
                ax.text(i - width, lr_vals[i] + 0.008, f'{lr_vals[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_CLM)
            if mlp_vals[i] > 0:
                ax.text(i, mlp_vals[i] + 0.008, f'{mlp_vals[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_PLM)
            ax.text(i + width, lora_vals[i] + 0.008, f'{lora_vals[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_LORA)

            if lr_vals[i] > 0:
                gap = lora_vals[i] - lr_vals[i]
                ax.text(i, max(lora_vals[i], mlp_vals[i]) + 0.035, f'+{gap:.2f}', ha='center', fontsize=5, fontweight='bold', color='#333333')

        ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, fontsize=5.5)
        ax.set_ylabel('ROC-AUC (test set)')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_ylim(0.45, 1.02)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=5, loc='upper left')

    fig.suptitle('Evaluation Protocol Determines Apparent Model Capability', fontsize=8, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig9_lora_hierarchy.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig9 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 11: 5-fold CV AUC box plot (replaces t-SNE)
# ═══════════════════════════════════════════════════════════════════

def make_fig11_cv_boxplot(data):
    comp = data['comp']
    esm2_t2 = data['esm2_t2']
    esm2_t3 = data['esm2_t3']
    esm1b = data['esm1b']
    encodon620m_t2 = data['encodon620m_t2']
    encodon620m_t3 = data['encodon620m_t3']
    codontransformer = data['codontransformer']
    calm = data['calm']

    clm_models = [
        ('encodon-80m', 'EnCodon-80M', comp),
        ('codonbert', 'CodonBERT', comp),
        ('codonbert_hf', 'CodonBERT-HF', comp),
    ]
    clm_ext = [
        ('encodon-620m', 'EnCodon-620M', [encodon620m_t2, encodon620m_t3]),
        ('codontransformer', 'CodonTransformer', codontransformer),
        ('calm', 'CaLM', calm),
    ]

    task2_folds = {}
    task3_folds = {}

    for internal, display, src in clm_models:
        for m in src:
            if m['model'] == internal:
                if m['task'] == 'task2_missense' and 'auc_folds' in m:
                    task2_folds[display] = m['auc_folds']
                elif m['task'] == 'task3_synonymous' and 'auc_folds' in m:
                    task3_folds[display] = m['auc_folds']

    for internal, display, src in clm_ext:
        if isinstance(src, list) and len(src) == 2 and isinstance(src[0], dict):
            t2 = src[0] if src[0].get('task') == 'task2_missense' else src[1]
            t3 = src[1] if src[1].get('task') == 'task3_synonymous' else src[0]
            if 'lr_auc_folds' in t2:
                task2_folds[display] = t2['lr_auc_folds']
            if 'lr_auc_folds' in t3:
                task3_folds[display] = t3['lr_auc_folds']
        elif isinstance(src, list):
            for entry in src:
                if entry.get('model', '').lower() == internal or entry.get('model') == display:
                    if entry.get('task') == 'task2_missense' and 'lr_auc_folds' in entry:
                        task2_folds[display] = entry['lr_auc_folds']
                    elif entry.get('task') == 'task3_synonymous' and 'lr_auc_folds' in entry:
                        task3_folds[display] = entry['lr_auc_folds']

    task2_folds['ESM-2'] = esm2_t2.get('ROC-AUC_folds', [esm2_t2['ROC-AUC_mean']] * 5)
    task3_folds['ESM-2'] = esm2_t3.get('ROC-AUC_folds', [esm2_t3['ROC-AUC_mean']] * 5)
    task2_folds['ESM-1b'] = esm1b['task2_missense'].get('ROC-AUC_folds', [esm1b['task2_missense']['ROC-AUC_mean']] * 5)
    task3_folds['ESM-1b'] = esm1b['task3_synonymous'].get('ROC-AUC_folds', [esm1b['task3_synonymous']['ROC-AUC_mean']] * 5)

    model_order = ['EnCodon-620M', 'CodonBERT', 'CodonTransformer', 'CodonBERT-HF', 'EnCodon-80M', 'CaLM', 'ESM-2', 'ESM-1b']
    type_colors = {'EnCodon-620M': '#92C5DE', 'CodonBERT': '#3B7DD8', 'CodonTransformer': '#5AAE61',
                   'CodonBERT-HF': '#6BA3E6', 'EnCodon-80M': '#9DC4F0', 'CaLM': '#A6D854',
                   'ESM-2': '#D6604D', 'ESM-1b': '#F4A582'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, folds_dict, title in zip(
        axes, [task2_folds, task3_folds],
        ['Task 1: Missense', 'Task 2: Synonymous']
    ):
        positions = []
        box_data = []
        colors = []
        labels_used = []
        for i, name in enumerate(model_order):
            if name in folds_dict:
                positions.append(i)
                box_data.append(folds_dict[name])
                colors.append(type_colors.get(name, GRAY))
                labels_used.append(name)

        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=True, notch=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_used, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('AUC (5-fold CV)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=8)

        for i, (name, d) in enumerate(zip(labels_used, box_data)):
            mean_val = np.mean(d)
            ax.text(positions[i], max(d) + 0.01, f'{mean_val:.3f}', ha='center', fontsize=6, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#3B7DD8', edgecolor='black', alpha=0.7, label='cLM (codon-BERT)'),
        mpatches.Patch(facecolor='#92C5DE', edgecolor='black', alpha=0.7, label='cLM (codon-EnCodon)'),
        mpatches.Patch(facecolor='#5AAE61', edgecolor='black', alpha=0.7, label='cLM (aa-codon)'),
        mpatches.Patch(facecolor='#D6604D', edgecolor='black', alpha=0.7, label='pLM'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('5-Fold Cross-Validation AUC Distribution', fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])

    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig11_cv_boxplot.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig11_cv_boxplot saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 7: Case Study + Attention
# ═══════════════════════════════════════════════════════════════════

def make_fig7_case_study():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    cases = ['PAH\nc.1197A>T', 'KCNQ1\nc.1032G>A', 'BRCA2\nc.9117G>A', 'MSH2\nc.2634G>A',
             'F8\nc.5217C>T', 'CFTR\nc.2988G>A', 'LDLR\nc.1216C>A', 'MLH1\nc.1038G>A',
             'APC\nc.1956C>T', 'ATM\nc.3576G>A']
    cb_mlp = [0.9998, 0.997, 0.993, 0.985, 0.976, 0.925, 0.983, 0.9997, 0.992, 0.980]
    enc_mlp = [0.934, 0.998, 0.983, 0.952, 0.993, 0.978, 0.985, 0.941, 0.945, 0.928]

    ax = axes[0]
    x_pos = np.arange(len(cases))
    width = 0.35
    ax.bar(x_pos - width/2, cb_mlp, width, label='CodonBERT', color=C_CLM, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + width/2, enc_mlp, width, label='EnCodon-80M', color=C_CLM3, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cases, fontsize=4.5, ha='center')
    ax.set_ylabel('P(pathogenic)')
    ax.set_ylim(0.8, 1.02)
    ax.set_title('MLP Probing: Pathogenic Synonymous Variants', fontsize=6.5, fontweight='bold')
    ax.legend(fontsize=5, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    attn_path = os.path.join(RESULTS_DIR, 'supplementary', 'codonbert_PAH_c.1197A_T_attention.npy')
    if os.path.exists(attn_path):
        attn = np.load(attn_path)
        avg_attn = attn.mean(axis=(0, 1))
        center = avg_attn.shape[0] // 2
        start = max(0, center - 8)
        end = min(avg_attn.shape[0], center + 9)
        local = avg_attn[start:end, start:end]
        im = ax.imshow(local, cmap='YlOrRd', aspect='equal', interpolation='nearest')
        ax.set_xlabel('Token Position (relative to center)')
        ax.set_ylabel('Token Position (relative to center)')
        n = end - start
        ticks = list(range(n))
        labels = [str(i - (center - start)) for i in range(n)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=5)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=5)
        ax.set_title('Attention: PAH c.1197A>T\n(CodonBERT, avg over layers/heads)', fontsize=6.5, fontweight='bold')
        center_local = center - start
        ax.add_patch(plt.Rectangle((center_local-0.5, center_local-0.5), 1, 1,
                     fill=False, edgecolor='black', linewidth=1.5, linestyle='--'))
    else:
        ax.text(0.5, 0.5, 'Attention data\nnot available', ha='center', va='center', fontsize=8)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig7_case_study.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig7 saved.")


# ═══════════════════════════════════════════════════════════════════
# FIG 8: Agent + Regression
# ═══════════════════════════════════════════════════════════════════

def make_fig8_regression():
    fig, ax = plt.subplots(figsize=(7, 3.5))

    models_reg = ['CodonBERT-HF', 'CodonBERT', 'EnCodon-80M']
    task3_r2 = [0.458, 0.321, 0.201]
    task3_sp = [0.684, 0.603, 0.614]
    task4_sp_ecoli = [0.252, 0.222, 0.192]
    task6_r2_fungal = [0.563, 0.544, 0.483]
    task6_sp_fungal = [0.734, 0.710, 0.701]
    task5_sp_mrna = [0.310, 0.264, 0.244]

    x_pos = np.arange(len(models_reg))
    width = 0.12
    bars1 = ax.bar(x_pos - 2.5*width, task3_r2, width, label='Task 3 R\u00b2 (within-protein)', color=C_CLM, alpha=0.85, edgecolor='white', linewidth=0.3)
    bars2 = ax.bar(x_pos - 1.5*width, task3_sp, width, label='Task 3 \u03c1', color=C_CLM2, alpha=0.85, edgecolor='white', linewidth=0.3)
    bars3 = ax.bar(x_pos - 0.5*width, task6_r2_fungal, width, label='Task 6 R\u00b2 (fungal cross-protein)', color='#2ca02c', alpha=0.85, edgecolor='white', linewidth=0.3)
    bars4 = ax.bar(x_pos + 0.5*width, task6_sp_fungal, width, label='Task 6 \u03c1', color='#98df8a', alpha=0.85, edgecolor='white', linewidth=0.3)
    bars5 = ax.bar(x_pos + 1.5*width, task4_sp_ecoli, width, label='Task 4 \u03c1 (E.coli cross-protein)', color=C_DNA, alpha=0.75, edgecolor='white', linewidth=0.3)
    bars6 = ax.bar(x_pos + 2.5*width, task5_sp_mrna, width, label='Task 5 \u03c1 (mRNA stability)', color=C_TRAD, alpha=0.75, edgecolor='white', linewidth=0.3)

    for bars, vals in [(bars1, task3_r2), (bars2, task3_sp), (bars3, task6_r2_fungal), (bars4, task6_sp_fungal), (bars5, task4_sp_ecoli), (bars6, task5_sp_mrna)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.012, f'{val:.3f}', ha='center', fontsize=4.5, fontweight='bold')

    ax.axhline(y=0, color='#BBBBBB', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_reg, fontsize=8)
    ax.set_ylabel('Score', fontsize=9)
    ax.set_ylim(-0.15, 0.85)
    ax.set_title('Regression Boundary of Codon-Level Predictability', fontsize=10, fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper right', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate('cLMs applicable\n(within-protein &\nfungal cross-protein)', xy=(0.5, 0.60), fontsize=6.5, fontstyle='italic', color='#2ca02c', ha='center')
    ax.annotate('cLMs limited\n(E.coli cross-protein\n& mRNA stability)', xy=(0.5, 0.08), fontsize=6.5, fontstyle='italic', color=C_TRAD, ha='center')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig8_regression_boundary.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig8_regression saved.")


def make_fig_agent():
    fig, ax = plt.subplots(figsize=(5, 3))

    dims = ['Execution\nTime', 'Error\nDetection', 'Reproducibility', 'Recommendation\nAccuracy']
    agent_scores = [30.8, 100, 100, 100]
    manual_scores = [1, 50, 30, 0]

    x_pos = np.arange(len(dims))
    width = 0.35
    ax.bar(x_pos - width/2, agent_scores, width, label='Agent', color=C_CLM, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + width/2, manual_scores, width, label='Manual', color=C_TRAD, alpha=0.6, edgecolor='white', linewidth=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims, fontsize=7)
    ax.set_ylabel('Score (normalized)', fontsize=8)
    ax.set_title('CodonBench-Agent vs Manual Evaluation', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0 - width/2, 30.8 + 2, '30.8\u00d7', ha='center', fontsize=7, fontweight='bold', color=C_CLM)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIG_DIR, f'fig_agent_comparison.{ext}'), dpi=300, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print("Fig_agent saved.")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading all data...")
    data = load_all_data()

    print("Building 21-model data...")
    model_auc, ESM2_T1, ESM2_T2, ESM1B_T1, ESM1B_T2 = build_21_models(data)
    print(f"  {len(model_auc)} models loaded")

    print("Building LoRA data...")
    lora_data = build_lora_data(data)

    print("Building CKA matrix...")
    cka_models, cka_lin, cka_rbf = build_cka_matrix(data)

    print("\nGenerating figures...")

    print("  Fig1 (overview schematic)...")
    make_fig1()

    print("  Fig2 (forest plot, 21 models)...")
    make_fig2(model_auc, data['bootstrap'], ESM2_T1, ESM2_T2)

    print("  Fig2_clm_vs_plm (cLM vs pLM crossover)...")
    make_fig2_clm_vs_plm(model_auc)

    print("  Fig3 (probing ablation)...")
    make_fig3_probing(lora_data)

    print("  Fig4 (LOGO recoverability)...")
    make_fig4(ESM2_T2)

    print("  Fig5 (CKA similarity, 6x6)...")
    make_fig5_cka(cka_models, cka_lin, cka_rbf)

    print("  Fig6_oracle (oracle ceiling)...")
    make_fig6_oracle(ESM1B_T2, ESM2_T2)

    print("  Fig11 (5-fold CV box plot)...")
    make_fig11_cv_boxplot(data)

    print("  Fig7 (case study)...")
    make_fig7_case_study()

    print("  Fig8 (regression boundary)...")
    make_fig8_regression()

    print("  Fig_agent (agent comparison, supplementary)...")
    make_fig_agent()

    print("  Fig9 (LoRA hierarchy, 5 cLMs)...")
    make_fig9_lora(lora_data)

    print(f"\nAll figures saved to {FIG_DIR}/")
