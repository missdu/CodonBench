# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
EXP-010 前置探测（纯 CPU，不占 GPU）
目的：回答"SynPath 的标签地基是否可靠"——Reject §二.4 + §十一.R4 最大方向性风险。

只做三件事：
1. 列出 synpath_variants.parquet 的全部列，定位 review status / 标签 / 基因 列；
2. 报 review status 的分布，并**按标签（良/恶）分层交叉**；
3. 与 e1s_usable / e1s_variant_index 对齐，确认主分析子集（n=2682）的分层是否偏。

不改动任何数据，不写结果文件（只打印）。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SYNPATH = BASE / "results" / "unified_eval" / "synpath_data"
OUT = BASE / "results" / "supplementary" / "wb_rerun"

print("=" * 70)
print("EXP-010 前置探测：SynPath 标签地基")
print("=" * 70)

# ---------- 1. 列清单 ----------
f = SYNPATH / "synpath_variants.parquet"
if not f.exists():
    print("[FATAL] 找不到 %s" % f)
    sys.exit(1)

df = pd.read_parquet(f)
print("\n[1] 形状: %d 行 x %d 列" % df.shape)
print("\n全部列：")
for c in df.columns:
    nn = df[c].notna().sum()
    print("   %-42s 非空 %6d  示例 %s" % (
        c, nn, str(df[c].dropna().iloc[0])[:38] if nn else "-"))

# ---------- 2. 定位关键列 ----------
def find(cands):
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        for lc, orig in low.items():
            if k in lc:
                return orig
    return None

col_rs = find(["review", "status", "star", "confidence", "gold"])
col_lab = find(["label", "class", "pathogen", "benign", "y"])
col_gene = find(["gene"])
col_trans = find(["transcript"])

print("\n[2] 关键列定位：")
print("    review status : %s" % col_rs)
print("    label         : %s" % col_lab)
print("    gene          : %s" % col_gene)
print("    transcript    : %s" % col_trans)

# ---------- 3. 标签与 review status 交叉 ----------
if col_lab:
    print("\n[3] 标签分布：")
    print(df[col_lab].value_counts(dropna=False).to_string())

if col_rs and col_lab:
    print("\n[4] review status × 标签 交叉表（这是地基风险的核心）：")
    ct = pd.crosstab(df[col_rs].fillna("<NA>"), df[col_lab].fillna("<NA>"),
                     margins=True)
    print(ct.to_string())

    # 致病组内高质量占比
    labs = df[col_lab].astype(str).str.lower()
    print("\n[5] 各标签组内 review status 构成（行内百分比）：")
    print(pd.crosstab(df[col_rs].fillna("<NA>"), df[col_lab].fillna("<NA>"),
                      normalize="columns").mul(100).round(1).to_string())

# ---------- 4. 与主分析子集对齐 ----------
try:
    usable = np.load(OUT / "e1s_usable.npy")
    genes = np.load(OUT / "e1s_genes.npy", allow_pickle=True)
    labels = np.load(OUT / "e1s_labels.npy")
    vidx = np.load(OUT / "e1s_variant_index.npy", allow_pickle=True)
    print("\n[6] 主分析子集：usable 条目 %d / 总 %d" % (int(usable.sum()), len(usable)))

    # variant_index 回源
    if len(vidx) == len(usable):
        sub_idx = np.array(vidx)[usable.astype(bool)]
        sub = df.iloc[sub_idx]
        print("    子集行数 %d" % len(sub))
        if col_rs and col_lab:
            print("\n[7] **主分析子集内** review status × 标签：")
            print(pd.crosstab(sub[col_rs].fillna("<NA>"),
                              sub[col_lab].fillna("<NA>"), margins=True).to_string())
            print("\n[8] **主分析子集内** 行内百分比：")
            print(pd.crosstab(sub[col_rs].fillna("<NA>"),
                              sub[col_lab].fillna("<NA>"),
                              normalize="columns").mul(100).round(1).to_string())
    else:
        print("    [warn] variant_index 长度 %d 与 usable %d 不匹配，无法回源"
              % (len(vidx), len(usable)))
except Exception as e:
    print("\n[warn] 与主分析子集对齐失败：%s" % e)

print("\n" + "=" * 70)
print("判读标准（先写死，避免事后解释）：")
print("  ① 若致病组几乎全是低星/单提交/无标准 ⇒ 标签噪声大，主结果须分层重跑；")
print("  ② 若良恶两组的 review status 构成**系统性不同** ⇒ 存在混杂，必须调整；")
print("  ③ 若高质量子集（≥2星或有标准）样本过少 ⇒ 分层分析只能作描述性，不得作主结果。")
print("=" * 70)
