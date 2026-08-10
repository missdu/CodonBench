import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import json
from pathlib import Path

OUT = Path('paper/figures_v2')
OUT.mkdir(exist_ok=True, parents=True)

# === Nature MI Style ===
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
    'figure.dpi': 300,
})

# Nature color palette
C_CLM = '#3B7DD8'
C_CLM2 = '#6BA3E6'
C_CLM3 = '#9DC4F0'
C_PLM = '#E64B35'
C_DNA = '#F39B30'
C_TRAD = '#666666'
C_TRAD2 = '#999999'
C_BENIGN = '#00A087'
C_PATHO = '#E64B35'

# ============================================================
# Fig 1: Benchmark Overview Schematic
# ============================================================
def fig1_overview():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Title
    ax.text(5, 3.8, 'CodonBench: Evaluation Framework', ha='center', va='top',
            fontsize=9, fontweight='bold')

    # Three columns: Models | Tasks | Evaluation
    cols = [(1.5, 'Models (12)'), (5, 'Tasks (5)'), (8.5, 'Evaluation Paradigms')]
    for x, label in cols:
        ax.text(x, 3.3, label, ha='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8E8', edgecolor='#999999', linewidth=0.5))

    # Models
    model_items = [
        (C_CLM, 'cLMs (3)\nCodonBERT\nEnCodon-80M\nCodonBERT-HF'),
        (C_PLM, 'pLM (1)\nESM-2-650M'),
        (C_DNA, 'DNA LM (2)\nNT-v2-50M\nNT-v2-500M'),
        (C_TRAD, 'Traditional (6)\none-hot, k-mer'),
    ]
    y_pos = 2.8
    for color, text in model_items:
        lines = text.split('\n')
        ax.text(1.5, y_pos, lines[0], ha='center', fontsize=5.5, fontweight='bold', color=color)
        for i, line in enumerate(lines[1:], 1):
            ax.text(1.5, y_pos - i*0.22, line, ha='center', fontsize=5, color='#444444')
        y_pos -= len(lines)*0.22 + 0.15

    # Tasks
    task_items = [
        ('Classification', 'Task 1: Missense\nTask 2: Synonymous', C_PATHO),
        ('Regression', 'Task 3: mRFP (within-protein)\nTask 4: E.coli (cross-protein)\nTask 5: mRNA stability', C_DNA),
    ]
    y_pos = 2.8
    for header, detail, color in task_items:
        ax.text(5, y_pos, header, ha='center', fontsize=5.5, fontweight='bold', color=color)
        for i, line in enumerate(detail.split('\n')):
            ax.text(5, y_pos - (i+1)*0.22, line, ha='center', fontsize=5, color='#444444')
        y_pos -= (len(detail.split('\n'))+1)*0.22 + 0.3

    # Evaluation paradigms
    eval_items = ['Level 0: Zero-shot LLR', 'Level 1: Linear probing', 'Level 2: Nonlinear probing (MLP)', 'Level 3: LoRA fine-tuning (r=8)']
    for i, item in enumerate(eval_items):
        alpha = 1.0 - i*0.15
        ax.text(8.5, 2.8 - i*0.4, item, ha='center', fontsize=5.5, color='#444444', alpha=alpha)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='#999999', linewidth=0.8)
    ax.annotate('', xy=(3.3, 1.8), xytext=(2.5, 1.8), arrowprops=arrow_props)
    ax.annotate('', xy=(6.8, 1.8), xytext=(6.0, 1.8), arrowprops=arrow_props)

    # Info decomposition at bottom
    ax.text(5, 0.3, r'$I(\mathrm{CDS}; Y) = I(A; Y) + I(\sigma; Y|A)$',
            ha='center', fontsize=7, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9E6', edgecolor='#CC9900', linewidth=0.5))
    ax.text(3.2, 0.1, '← pLM captures', ha='center', fontsize=5, color=C_PLM)
    ax.text(6.8, 0.1, 'cLM captures →', ha='center', fontsize=5, color=C_CLM)

    fig.savefig(OUT / 'fig1_overview.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig1_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig1")

# ============================================================
# Fig 2: cLM vs pLM Complementary Strengths (THE core figure)
# ============================================================
def fig2_clm_vs_plm():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    models = ['ESM-2-650M', 'CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']
    types = ['pLM', 'cLM', 'cLM', 'cLM']
    colors = [C_PLM, C_CLM, C_CLM2, C_CLM3]

    # Task 1: Missense
    t2_auc = [0.719, 0.660, 0.659, 0.633]
    t2_std = [0.016, 0.014, 0.014, 0.023]
    t2_ci = [[0.706, 0.734], [0.648, 0.672], [0.648, 0.671], [0.613, 0.653]]

    # Task 2: Synonymous
    t3_auc = [0.680, 0.734, 0.705, 0.686]
    t3_std = [0.023, 0.007, 0.013, 0.030]
    t3_ci = [[0.660, 0.699], [0.727, 0.739], [0.693, 0.716], [0.665, 0.715]]

    for ax_idx, (ax, aucs, stds, cis, title) in enumerate(zip(
        axes, [t2_auc, t3_auc], [t2_std, t3_std], [t2_ci, t3_ci],
        ['Task 1: Missense Variant\nPathogenicity', 'Task 2: Synonymous Variant\nPathogenicity'])):

        y_pos = np.arange(len(models))
        bars = ax.barh(y_pos, aucs, height=0.6, color=colors, edgecolor='white', linewidth=0.3, alpha=0.85)

        # Error bars from bootstrap CI
        for i, (auc, ci) in enumerate(zip(aucs, cis)):
            ax.plot([ci[0], ci[1]], [i, i], color='#333333', linewidth=0.8, solid_capstyle='butt')
            ax.plot(ci[0], i, '|', color='#333333', markersize=3)
            ax.plot(ci[1], i, '|', color='#333333', markersize=3)

        ax.axvline(x=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=6)
        ax.set_xlabel('ROC-AUC')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_xlim(0.45, 0.78)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for i, auc in enumerate(aucs):
            ax.text(auc + 0.008, i, f'{auc:.3f}', va='center', fontsize=5.5, fontweight='bold')

    # Add type labels
    for ax in axes:
        ax.text(-0.02, -0.15, 'pLM', transform=ax.get_yaxis_transform(),
                fontsize=5.5, color=C_PLM, fontweight='bold', ha='right')
        ax.text(-0.02, 0.5, 'cLMs', transform=ax.get_yaxis_transform(),
                fontsize=5.5, color=C_CLM, fontweight='bold', ha='right')

    # Add crossover annotation
    fig.text(0.5, -0.02, '← pLM wins          cLMs win →', ha='center', fontsize=5.5,
             fontstyle='italic', color='#666666')

    fig.tight_layout()
    fig.savefig(OUT / 'fig2_clm_vs_plm.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig2_clm_vs_plm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig2")

# ============================================================
# Fig 3: Probing Method Ablation
# ============================================================
def fig3_probing_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    # Left: CodonBERT-HF Task 2 ablation
    methods = ['LR (C=0.01)', 'KNN-5', 'LR (C=100)', 'LR (C=1.0)', 'MLP (128→64)']
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

    # Annotate the gap
    ax.annotate('', xy=(0.867, 4.3), xytext=(0.705, 3.7),
                arrowprops=dict(arrowstyle='<->', color=C_PLM, linewidth=1.0))
    ax.text(0.82, 4.6, '+23%', fontsize=6, fontweight='bold', color=C_PLM, ha='center')

    # Right: LR vs MLP vs LoRA across all models (Task 2)
    ax = axes[1]
    models_mlp = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']
    lr_aucs = [0.716, 0.668, 0.687]
    mlp_aucs = [0.849, 0.816, 0.812]
    lora_aucs = [0.912, 0.880, 0.893]

    x_pos = np.arange(len(models_mlp))
    width = 0.25
    ax.bar(x_pos - width, lr_aucs, width, label='LR (Level 1)', color=C_CLM, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos, mlp_aucs, width, label='MLP (Level 2)', color=C_PLM, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + width, lora_aucs, width, label='LoRA (Level 3)', color='#7B2D8E', alpha=0.7, edgecolor='white', linewidth=0.3)

    for i, (lr, mlp, lora) in enumerate(zip(lr_aucs, mlp_aucs, lora_aucs)):
        ax.text(i - width, lr + 0.01, f'{lr:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_CLM)
        ax.text(i, mlp + 0.01, f'{mlp:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_PLM)
        ax.text(i + width, lora + 0.01, f'{lora:.3f}', ha='center', fontsize=4.5, fontweight='bold', color='#7B2D8E')
        pct = (lora - lr) / lr * 100
        ax.text(i, lora + 0.04, f'+{pct:.0f}%\nvs LR', ha='center', fontsize=4.5, fontweight='bold', color='#7B2D8E')

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
    fig.savefig(OUT / 'fig3_probing_ablation.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig3_probing_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig3")

# ============================================================
# Fig 4: Comprehensive Model Comparison (dot plot)
# ============================================================
def fig4_comprehensive():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # Task 1
    t2_data = [
        ('onehot_pos', 'Traditional', 0.755, 0.011),
        ('ESM-2-650M', 'pLM', 0.719, 0.016),
        ('onehot_freq', 'Traditional', 0.667, 0.014),
        ('combined', 'Traditional', 0.666, 0.017),
        ('CodonBERT', 'cLM', 0.660, 0.014),
        ('CodonBERT-HF', 'cLM', 0.659, 0.014),
        ('kmer6', 'Traditional', 0.651, 0.025),
        ('EnCodon-80M', 'cLM', 0.633, 0.023),
        ('kmer4', 'Traditional', 0.627, 0.019),
        ('kmer3', 'Traditional', 0.621, 0.020),
        ('NT-50M', 'DNA LM', 0.573, 0.018),
        ('NT-500M', 'DNA LM', 0.570, 0.021),
    ]

    t3_data = [
        ('onehot_pos', 'Traditional', 0.891, 0.021),
        ('CodonBERT', 'cLM', 0.734, 0.007),
        ('kmer6', 'Traditional', 0.700, 0.029),
        ('NT-500M', 'DNA LM', 0.699, 0.008),
        ('CodonBERT-HF', 'cLM', 0.705, 0.013),
        ('EnCodon-80M', 'cLM', 0.686, 0.030),
        ('ESM-2-650M', 'pLM', 0.680, 0.023),
        ('NT-50M', 'DNA LM', 0.643, 0.019),
        ('combined', 'Traditional', 0.594, 0.013),
        ('onehot_freq', 'Traditional', 0.589, 0.010),
        ('kmer4', 'Traditional', 0.580, 0.024),
        ('kmer3', 'Traditional', 0.562, 0.021),
    ]

    color_map = {'cLM': C_CLM, 'pLM': C_PLM, 'DNA LM': C_DNA, 'Traditional': C_TRAD}
    marker_map = {'cLM': 'o', 'pLM': 's', 'DNA LM': 'D', 'Traditional': '^'}

    for ax, data, title in zip(axes, [t2_data, t3_data],
            ['Task 1: Missense\nVariant Pathogenicity', 'Task 2: Synonymous\nVariant Pathogenicity']):

        for name, mtype, auc, std in data:
            y = auc
            ax.errorbar(auc, name, xerr=std, fmt=marker_map[mtype], color=color_map[mtype],
                       markersize=4, capsize=2, capthick=0.5, elinewidth=0.5, alpha=0.85)

        ax.axvline(x=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.set_xlabel('ROC-AUC')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_xlim(0.45, 0.95)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=5.5)

    # Legend
    legend_elements = [plt.Line2D([0], [0], marker=marker_map[k], color=color_map[k],
                       label=k, markersize=4, linestyle='None') for k in ['cLM', 'pLM', 'DNA LM', 'Traditional']]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=5)

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_comprehensive.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig4_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig4")

# ============================================================
# Fig 5: CKA Similarity Matrix
# ============================================================
def fig5_cka():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))

    models = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']

    # Linear CKA
    lin_cka = np.array([
        [1.000, 0.007, 0.009],
        [0.007, 1.000, 0.009],
        [0.009, 0.009, 1.000],
    ])

    # RBF CKA
    rbf_cka = np.array([
        [1.000, 0.052, 0.211],
        [0.052, 1.000, 0.219],
        [0.211, 0.219, 1.000],
    ])

    for ax, data, title in zip(axes, [lin_cka, rbf_cka], ['Linear CKA', 'RBF CKA']):
        im = ax.imshow(data, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        ax.set_xticks(range(3))
        ax.set_xticklabels(models, fontsize=5.5, rotation=30, ha='right')
        ax.set_yticks(range(3))
        ax.set_yticklabels(models, fontsize=5.5)
        ax.set_title(title, fontsize=7, fontweight='bold')

        for i in range(3):
            for j in range(3):
                val = data[i, j]
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6, fontweight='bold', color=color)

        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=5)

    fig.suptitle('Representation Similarity Between cLMs (Task 2)', fontsize=7, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_cka_similarity.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig5_cka_similarity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig5")

# ============================================================
# Fig 6: Embedding t-SNE Visualization
# ============================================================
def fig6_tsne():
    sup_dir = Path('results/supplementary')

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    models = ['codonbert', 'codonbert_hf', 'encodon-80m']
    tasks = ['task2_missense', 'task3_synonymous']
    model_labels = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']
    task_labels = ['Task 1: Missense', 'Task 2: Synonymous']

    for row, (task, task_label) in enumerate(zip(tasks, task_labels)):
        for col, (model, model_label) in enumerate(zip(models, model_labels)):
            ax = axes[row, col]
            tsne_path = sup_dir / f'{model}_{task}_tsne.npy'
            lab_path = sup_dir / f'{model}_{task}_tsne_labels.npy'

            if tsne_path.exists() and lab_path.exists():
                coords = np.load(tsne_path)
                labels = np.load(lab_path)

                mask_0 = labels == 0
                mask_1 = labels == 1

                ax.scatter(coords[mask_0, 0], coords[mask_0, 1], s=2, alpha=0.3, color=C_BENIGN, label='Benign', rasterized=True)
                ax.scatter(coords[mask_1, 0], coords[mask_1, 1], s=2, alpha=0.3, color=C_PATHO, label='Pathogenic', rasterized=True)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=8, color='#999999')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if row == 0:
                ax.set_title(model_label, fontsize=6.5, fontweight='bold')
            if col == 0:
                ax.set_ylabel(task_label, fontsize=6)

    # Single legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=C_BENIGN, label='Benign', markersize=4, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color=C_PATHO, label='Pathogenic', markersize=4, linestyle='None'),
    ]
    axes[0, 2].legend(handles=legend_elements, loc='upper right', fontsize=5, framealpha=0.8)

    fig.suptitle('t-SNE Visualization of cLM Embeddings', fontsize=7, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_tsne_embeddings.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig6_tsne_embeddings.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig6")

# ============================================================
# Fig 7: Case Study + Attention
# ============================================================
def fig7_case_study():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Left: Case study prediction probabilities
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

    # Right: Attention heatmap for one example
    ax = axes[1]
    attn_path = Path('results/supplementary/codonbert_PAH_c.1197A>T_attention.npy')
    if attn_path.exists():
        attn = np.load(attn_path)  # (n_layers, n_heads, seq_len, seq_len)
        # Average over layers and heads
        avg_attn = attn.mean(axis=(0, 1))  # (seq_len, seq_len)
        # Show center ±8 positions
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

        # Mark center
        center_local = center - start
        ax.add_patch(plt.Rectangle((center_local-0.5, center_local-0.5), 1, 1,
                     fill=False, edgecolor='black', linewidth=1.5, linestyle='--'))
    else:
        ax.text(0.5, 0.5, 'Attention data\nnot available', ha='center', va='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / 'fig7_case_study.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig7_case_study.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig7")

# ============================================================
# Fig 8: Agent Framework + Regression Boundary
# ============================================================
def fig8_agent_regression():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # Left: Agent comparison
    ax = axes[0]
    dims = ['Execution\nTime', 'Error\nDetection', 'Reproducibility', 'Recommendation\nAccuracy']
    agent_scores = [30.8, 100, 100, 100]
    manual_scores = [1, 50, 30, 0]  # normalized for comparison

    x_pos = np.arange(len(dims))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, agent_scores, width, label='Agent', color=C_CLM, alpha=0.8, edgecolor='white', linewidth=0.3)
    bars2 = ax.bar(x_pos + width/2, manual_scores, width, label='Manual', color=C_TRAD, alpha=0.6, edgecolor='white', linewidth=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dims, fontsize=5.5)
    ax.set_ylabel('Score (normalized)')
    ax.set_title('CodonBench-Agent\nvs Manual Evaluation', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5.5, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate 30.8x
    ax.text(0 - width/2, 30.8 + 2, '30.8×', ha='center', fontsize=5.5, fontweight='bold', color=C_CLM)

    # Right: Regression boundary
    ax = axes[1]
    models_reg = ['CodonBERT-HF', 'CodonBERT', 'EnCodon-80M']
    task4_r2 = [0.458, 0.321, 0.201]
    task4_sp = [0.684, 0.603, 0.614]
    task5_sp_ecoli = [0.252, 0.222, 0.192]
    task5_sp_mrna = [0.310, 0.264, 0.244]

    x_pos = np.arange(len(models_reg))
    width = 0.2
    ax.bar(x_pos - 1.5*width, task4_r2, width, label='Task 3 R² (within-protein)', color=C_CLM, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos - 0.5*width, task4_sp, width, label='Task 3 ρ', color=C_CLM2, alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + 0.5*width, task5_sp_ecoli, width, label='Task 4 ρ (cross-protein)', color=C_DNA, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.bar(x_pos + 1.5*width, task5_sp_mrna, width, label='Task 5 ρ (cross-transcript)', color=C_TRAD, alpha=0.7, edgecolor='white', linewidth=0.3)

    ax.axhline(y=0, color='#BBBBBB', linestyle='--', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_reg, fontsize=6)
    ax.set_ylabel('Score')
    ax.set_ylim(-0.1, 0.8)
    ax.set_title('Regression: Boundary of\nCodon-Level Predictability', fontsize=7, fontweight='bold')
    ax.legend(fontsize=4.5, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate boundary
    ax.annotate('cLMs applicable', xy=(0.5, 0.55), fontsize=5, fontstyle='italic', color=C_CLM, ha='center')
    ax.annotate('cLMs limited', xy=(0.5, 0.15), fontsize=5, fontstyle='italic', color=C_TRAD, ha='center')

    fig.tight_layout()
    fig.savefig(OUT / 'fig8_agent_regression.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig8_agent_regression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig8")

# ============================================================
# Fig 9: LoRA Evaluation Hierarchy (Finding 3 core figure)
# ============================================================
def fig9_lora_hierarchy():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    C_LORA = '#7B2D8E'

    models = ['CodonBERT', 'CodonBERT-HF', 'EnCodon-80M']

    # Task 1
    t2_lr = [0.662, 0.646, 0.636]
    t2_mlp = [0.677, 0.650, 0.674]
    t2_lora = [0.808, 0.814, 0.696]

    # Task 2
    t3_lr = [0.716, 0.668, 0.687]
    t3_mlp = [0.849, 0.816, 0.812]
    t3_lora = [0.912, 0.880, 0.893]

    for ax, lr, mlp, lora, title in zip(
        axes,
        [t2_lr, t3_lr], [t2_mlp, t3_mlp], [t2_lora, t3_lora],
        ['Task 1: Missense\nVariant Pathogenicity', 'Task 2: Synonymous\nVariant Pathogenicity']
    ):
        x_pos = np.arange(len(models))
        width = 0.22

        bars_lr = ax.bar(x_pos - width, lr, width, label='LR (Level 1)', color=C_CLM, alpha=0.75, edgecolor='white', linewidth=0.3)
        bars_mlp = ax.bar(x_pos, mlp, width, label='MLP (Level 2)', color=C_PLM, alpha=0.75, edgecolor='white', linewidth=0.3)
        bars_lora = ax.bar(x_pos + width, lora, width, label='LoRA (Level 3)', color=C_LORA, alpha=0.75, edgecolor='white', linewidth=0.3)

        for i in range(len(models)):
            ax.text(i - width, lr[i] + 0.008, f'{lr[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_CLM)
            ax.text(i, mlp[i] + 0.008, f'{mlp[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_PLM)
            ax.text(i + width, lora[i] + 0.008, f'{lora[i]:.3f}', ha='center', fontsize=4.5, fontweight='bold', color=C_LORA)

            gap = lora[i] - lr[i]
            ax.annotate('', xy=(i + width + 0.05, lora[i]), xytext=(i - width - 0.05, lr[i]),
                        arrowprops=dict(arrowstyle='<->', color='#333333', linewidth=0.6))
            ax.text(i, max(lora[i], mlp[i]) + 0.035, f'+{gap:.2f}', ha='center', fontsize=5, fontweight='bold', color='#333333')

        ax.axhline(y=0.5, color='#BBBBBB', linestyle='--', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=6)
        ax.set_ylabel('ROC-AUC (test set)')
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.set_ylim(0.45, 1.02)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].legend(fontsize=5, loc='upper left')

    fig.suptitle('Evaluation Protocol Determines Apparent Model Capability', fontsize=8, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig9_lora_hierarchy.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig9_lora_hierarchy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved fig9")


# ============================================================
# Generate all figures
# ============================================================
if __name__ == '__main__':
    print("Generating Nature MI-style figures...")
    fig1_overview()
    fig2_clm_vs_plm()
    fig3_probing_ablation()
    fig4_comprehensive()
    fig5_cka()
    fig6_tsne()
    fig7_case_study()
    fig8_agent_regression()
    fig9_lora_hierarchy()
    print(f"\nAll 9 figures saved to {OUT}/")