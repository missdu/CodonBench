"""
Generate a bilingual reading version of v11 paper.
For each paragraph, show English original + Chinese translation.
Also annotate each figure with its data source file.
"""
import re
import os

MD_PATH = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\codonbench_paper_v11.md'
OUT_PATH = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\paper\codonbench_paper_v11_bilingual.md'
RESULTS_DIR = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results'

with open(MD_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# Figure data source mapping
FIG_DATA = {
    'fig1_overview_v2': '概念图，无原始数据文件。内容来自§3.1信息分解框架。',
    'fig2_forest_plot': f'原始数据: {RESULTS_DIR}\\comprehensive_comparison.json + {RESULTS_DIR}\\bootstrap_ci_results.json',
    'fig2_clm_vs_plm': f'原始数据: {RESULTS_DIR}\\comprehensive_comparison.json (Task 1 + Task 2 AUC)',
    'fig4_logo_recoverability': f'原始数据: {RESULTS_DIR}\\logo_cv_results.json',
    'fig3_probing_ablation': f'原始数据: {RESULTS_DIR}\\probing_ablation_codonbert_task2.json + {RESULTS_DIR}\\probing_ablation_codonbert_hf_task3.json',
    'fig9_lora_hierarchy': f'原始数据: {RESULTS_DIR}\\lora_finetune_results.json + {RESULTS_DIR}\\lora_extended_results.json',
    'fig6_oracle_ceiling': f'原始数据: {RESULTS_DIR}\\lora_extended_results.json (CodonTransformer 0.943) + oracle probe结果',
    'fig5_cka_similarity': f'原始数据: {RESULTS_DIR}\\cka_revalidation.json + {RESULTS_DIR}\\cka_extended_results.json',
    'fig7_context': f'原始数据: {RESULTS_DIR}\\downstream_cds_full_v2_summary.json (real CDS) + {RESULTS_DIR}\\downstream_classification_summary_final.json (synthetic)',
    'fig8_agent_regression': f'回归数据: {RESULTS_DIR}\\regression_summary.json; Agent数据: {RESULTS_DIR}\\agent_vs_manual.json',
    'fig9_interpretability': f'原始数据: {RESULTS_DIR}\\biological_findings_results.json',
    'fig7_case_study': f'原始数据: {RESULTS_DIR}\\case_study_results.json',
    'fig6_tsne_embeddings': f'原始数据: {RESULTS_DIR}\\supplementary\\ (embedding .npy files)',
}

# Chinese translations for key sections (manual, high-quality)
CN_SECTIONS = {
    'Abstract': '摘要\n\n同义密码子选择编码了蛋白质级模型无法看到的调控信息——然而设计用于捕获该信号的密码子语言模型(cLM)仍无统一评估框架：自2023年以来已发表30余个，但碎片化协议和85%不可访问率导致零个可直接比较。我们提出CodonBench，基于信息论恒等式I(CDS;Y)=I(A;Y)+I(σ;Y|A)，将编码序列信息分解为蛋白质信道和同义信道。我们首先以模型无关方式证明同义信道携带真实信号：无需预训练的密码子6-mer在同义致病变异上达AUC 0.700。\n\n评估21个模型×5任务×4级协议，发现可机制解释的交叉反转：同义变异上密码子级cLM超过pLM基线(ESM-2)达+0.105 AUC；错义变异上pLM领先所有cLM。该优势被分词策略门控。三个进一步发现：探测方法是混淆变量(+10-13pp)；LoRA微调达oracle天花板99.4%；跨蛋白表达R²<0划定适用边界。',

    '1. Introduction': '1. 引言\n\n遗传密码简并性——64个密码子编码20个氨基酸——创造了蛋白质序列之外的调控信息层。同义密码子选择影响mRNA稳定性、翻译延伸速率和共翻译折叠。关键的是，ClinVar中约10-15%的致病变异是同义的——这些改变保持氨基酸序列不变却破坏功能，对蛋白质级模型完全不可见。',

    '1.1': '1.1 密码子级信息的临床和生物技术意义\n\n密码子优化是mRNA疫苗设计的核心，同义选择直接影响表达产量和免疫原性。',

    '1.2': '1.2 密码子语言模型的评估危机\n\n30余个cLM已发表，但评估碎片化+标准化危机(仅15%可通过标准接口加载)，导致"30+模型发表，0个可直接比较"。这类似GLUE之前的NLP基准危机。',

    '1.3': '1.3 我们的方法和贡献\n\n三项贡献：(1)信息论框架+可证伪预测P1/P2；(2)七项实证发现重塑cLM评估；(3)20个cLM的生态可访问性审计。',

    'P1': 'P1(同义体制)：同义变异上A固定，I(A;Y)=0，若I(σ;Y|A)>0则cLM应优于pLM。可证伪：cLM未胜pLM则I(σ;Y|A)≈0。',
    'P2': 'P2(错义体制)：错义变异上I(A;Y)>0且主导，pLM应优于cLM。可证伪：pLM未胜cLM则I(σ;Y|A)在A改变时仍不可忽略。',

    'Finding1': '发现1：cLM-pLM互补性真实但以分词为条件\n\n同义变异：EnCodon-620M 0.785 > ESM-2 0.680 (Δ=+0.105)；错义变异：ESM-2 0.719 > 最佳cLM 0.694。关键：仅密码子级cLM访问I(σ;Y|A)，字符级和极小维度cLM不能。',

    'Finding2': '发现2：探测方法是混淆变量\n\nMLP比线性探测高10-13个百分点，系统性低估cLM。单一探测基准可反转模型排名。',

    'Finding3': '发现3：LoRA微调释放隐藏能力\n\nCodonTransformer达0.943，接近oracle天花板0.949的99.4%。评估协议差距>模型间差距。',

    'Finding4': '发现4：分词决定表示几何\n\nCKA层级：同架构(0.66) > 同分词(0.36-0.48) > 不同分词(0.31-0.44)。BERT族消融：密码子分词vs字符分词 Δ=+0.120。',

    'Finding5': '发现5：真实CDS上下文至关重要\n\n真实CDS比合成上下文提升18-26% AUC。合成上下文严重低估模型能力。',

    'Finding6': '发现6：密码子级预测有明确适用边界\n\n蛋白内表达R²=0.458(可用)，跨蛋白表达R²<0(不可部署)。',

    'Finding7': '发现7：cLM嵌入编码密码子使用偏好，非线性编码致病性\n\nGC3 R²=0.72-0.95(主特征)，CAI因分词/参考基因组而异，致病性需非线性提取。',

    'Limitations': '局限\n\n(1)分词vs容量未完全分离(需受控预训练)；(2)分类规模有限(1,420同义致病)；(3)仅ClinVar单数据集；(4)500转录本CDS覆盖有限；(5)预训练污染未完全排除。',
}

# Build bilingual document
lines = text.split('\n')
output_lines = []

output_lines.append('# CodonBench v11 论文 — 中英对照阅读版\n\n')

for i, line in enumerate(lines):
    stripped = line.strip()

    # Skip double --- separators
    if stripped == '---':
        output_lines.append('---\n')
        continue

    # Add figure data annotation
    img_match = re.match(r'!\[([^\]]+)\]\(figures_v2/([^\)]+)\)', stripped)
    if img_match:
        alt = img_match.group(1)
        png = img_match.group(2)
        base = os.path.splitext(png)[0]
        output_lines.append(line + '\n\n')
        if base in FIG_DATA:
            output_lines.append(f'> **📊 图数据来源**: {FIG_DATA[base]}\n\n')
        continue

    # Add Chinese translation after key section headers
    if stripped.startswith('## Abstract'):
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Abstract"]}\n\n')
        continue

    if stripped.startswith('## 1. Introduction'):
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["1. Introduction"]}\n\n')
        continue

    if stripped.startswith('### 1.1'):
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["1.1"]}\n\n')
        continue

    if stripped.startswith('### 1.2'):
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["1.2"]}\n\n')
        continue

    if stripped.startswith('### 1.3'):
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["1.3"]}\n\n')
        continue

    # Add Chinese for key findings
    if 'Finding 1:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding1"]}\n\n')
        continue

    if 'Finding 2:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding2"]}\n\n')
        continue

    if 'Finding 3:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding3"]}\n\n')
        continue

    if 'Finding 4:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding4"]}\n\n')
        continue

    if 'Finding 5:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding5"]}\n\n')
        continue

    if 'Finding 6:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding6"]}\n\n')
        continue

    if 'Finding 7:' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Finding7"]}\n\n')
        continue

    if 'P1 (synonymous' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["P1"]}\n\n')
        continue

    if 'P2 (missense' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["P2"]}\n\n')
        continue

    if '### 5.4 Limitations' in stripped:
        output_lines.append(line + '\n\n')
        output_lines.append(f'> **中文**：{CN_SECTIONS["Limitations"]}\n\n')
        continue

    output_lines.append(line + '\n')

# Append figure improvement suggestions
output_lines.append('\n---\n\n')
output_lines.append('# 图表改进建议\n\n')

suggestions = [
    ('Fig 1 (Overview)', 'fig1_overview_v2.png', 
     '当前：三栏概念图。问题：信息密度低，审稿人可能觉得"太概念化"。\n改进方案：(A) 加入真实数据热力图作为背景，如Task 1/2的AUC矩阵；(B) 用Sankey图展示信息流从CDS→A+σ→Y；(C) 在P1/P2栏加入实际AUC数值标注。'),
    ('Fig 2 (Forest plot)', 'fig2_forest_plot.png',
     '当前：传统forest plot。问题：21个模型×2任务=42个点，可能拥挤。\n改进方案：(A) 按tokenization类型分组用不同颜色/形状；(B) 加入ESM-2基线虚线；(C) 用raincloud plot替代，同时展示分布+点估计+CI。'),
    ('Fig 3 (cLM-pLM crossover)', 'fig2_clm_vs_plm.png',
     '当前：散点图展示cLM vs pLM。问题：可能不够直观。\n改进方案：(A) 用paired slope图（每个模型一条线连接Task1和Task2）；(B) 或用2D scatter with diagonal，颜色=tokenization类型。'),
    ('Fig 3b (LOGO recoverability)', 'fig4_logo_recoverability.png',
     '当前：标准CV vs LOGO-CV柱状图。问题：只展示4个模型。\n改进方案：(A) 加入LoRA恢复结果作为第三组柱子（当前只有文字描述）；(B) 用waterfall图展示Standard→LOGO→LoRA的AUC变化路径。'),
    ('Fig 4 (Probing ablation)', 'fig3_probing_ablation.png',
     '当前：7种探测方法消融。问题：只展示1-2个模型。\n改进方案：(A) 扩展到所有5个cLM，用热力图（行=模型，列=探测方法，色=AUC）；(B) 或用ridge plot展示AUC分布。'),
    ('Fig 5+5b (LoRA + Oracle)', 'fig9_lora_hierarchy.png + fig6_oracle_ceiling.png',
     '当前：两张独立图。问题：信息分散。\n改进方案：(A) 合并为一张图：左=LR→MLP→LoRA递进柱状图，右=oracle ceiling水平线+各模型LoRA点；(B) 用bullet chart（类似target图）展示每个cLM相对oracle的完成度。'),
    ('Fig 6 (CKA heatmap)', 'fig5_cka_similarity.png',
     '当前：热力图。问题：8对模型的热力图可能不够直观。\n改进方案：(A) 改为clustered heatmap with dendrogram；(B) 或用MDS/t-SNE降维到2D，点=模型，距离=1-CKA，颜色=tokenization类型。'),
    ('Fig 7 (Context ablation)', 'fig7_context.png',
     '当前：分组柱状图。问题：只展示Task 1。\n改进方案：(A) 加入zero-shot层面的对比数据（右面板已有但数据稀疏）；(B) 用dumbbell plot（左=合成，右=真实CDS，线=提升幅度）。'),
    ('Fig 8 (Regression + Agent)', 'fig8_agent_regression.png',
     '当前：双面板。问题：回归边界和agent加速是两个不相关的故事。\n改进方案：(A) 回归部分用scatter plot with regression line，展示R²；(B) agent部分用speedup bar + accuracy指标组合。'),
    ('Fig 9 (GC3/CAI)', 'fig9_interpretability.png',
     '当前：分组柱+pathogenicity柱。问题：R²为负的Position柱压缩了有效数据的可视范围。\n改进方案：(A) 拆分为两个独立面板，左=R²(只展示0-1范围)，右=pathogenicity AUC(zoomed)；(B) 或用radar/spider chart展示每个cLM的多维属性编码能力。'),
    ('Fig 10 (Case study)', 'fig7_case_study.png',
     '当前：15个变异的预测概率。问题：样本量小，视觉冲击力不足。\n改进方案：(A) 用waterfall/diverging bar展示P(pathogenic)-0.5；(B) 标注基因名+疾病名在bar旁边；(C) 加入ESM-2的预测作为对比。'),
    ('Fig 11 (t-SNE)', 'fig6_tsne_embeddings.png',
     '当前：t-SNE散点图。问题：t-SNE的cluster可能不反映真实结构。\n改进方案：(A) 同时展示UMAP对比；(B) 用convex hull圈出pathogenic vs benign区域；(C) 加入perplexity sensitivity分析。'),
]

for title, fname, suggestion in suggestions:
    output_lines.append(f'## {title}\n\n')
    output_lines.append(f'当前文件: `figures_v2/{fname}`\n\n')
    output_lines.append(f'{suggestion}\n\n')

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print(f'Written: {OUT_PATH}')
print(f'Lines: {len(output_lines)}')