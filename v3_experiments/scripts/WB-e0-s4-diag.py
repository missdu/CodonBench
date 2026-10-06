#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-s4-diag.py —— 诊断 S4 里那个极端不平衡：
    d_exon == 0 的变异里，致病组 774 条、良性组 1 条。

要回答三问：
  Q1 这些变异是落在外显子**第一个**碱基，还是**最后一个**碱基？
  Q2 相应外显子长度是多少（若大量长度为 1 或 2，说明结构解析有问题）？
  Q3 落在边缘的是不是集中在少数基因/少数变异（若是，说明是数据集构造导致，不是普遍现象）？

判据：若 Q2 显示外显子长度正常、且 Q1 显示集中在最后一位，
      而 Q3 显示分散在很多基因 ⇒ 这是**数据本身的富集**，可以写进 S4；
      若 Q3 显示集中在极少数基因 ⇒ 需进一步查是不是坐标口径问题。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import json
import numpy as np
import pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
OUT = f"{BASE}/results/supplementary/wb_rerun/wb_frozen"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
v2 = pd.read_csv(f"{OUT}/dataset_v2/variants.tsv", sep="\t", dtype=str)
rg = pd.read_csv(f"{BASE}/WB-refgene_exons.tsv", sep="\t",
                 dtype={"tx_acc": str, "chrom": str})
rg = rg[~rg["chrom"].str.contains("_", na=False)]

idx = {}
for row in rg.itertuples(index=False):
    idx.setdefault((row.tx_acc, row.build), row)

BUILD = {"GRCh37": "hg19", "GRCh38": "hg38"}
n = len(df)
y = df["label"].values.astype(int)
tx = np.array([str(x).split(".")[0] for x in df["tx_id"].values], dtype=object)
build = np.array([BUILD.get(str(a), "?") for a in df["Assembly"].values], dtype=object)
chrom = np.array(["chr" + str(c) for c in df["Chromosome"].values], dtype=object)
pos0 = df["PositionVCF"].values.astype(int) - 1
gene = df["gene_symbol"].values

rows = []
for i in range(n):
    row = idx.get((tx[i], build[i]))
    if row is None or row.chrom != chrom[i]:
        continue
    starts = [int(x) for x in str(row.exon_starts).split(",")]
    ends = [int(x) for x in str(row.exon_ends).split(",")]
    p = int(pos0[i])
    best, which, elen, off_in = None, None, None, None
    for k, (a, b) in enumerate(zip(starts, ends)):
        for tag, d in (("first", abs(p - a)), ("last", abs(p - (b - 1)))):
            if best is None or d < best:
                best, which, elen, off_in = d, tag, b - a, p - a
    rows.append((i, int(y[i]), str(gene[i]), str(tx[i]), best, which, elen, off_in,
                 int(row.n_exon)))

d = pd.DataFrame(rows, columns=["idx", "label", "gene", "tx", "d", "end",
                                "exon_len", "off_in_exon", "n_exon"])
print(f"解析到 {len(d)} 条")

z = d[d["d"] == 0]
print(f"\n=== Q1 d==0 的 {len(z)} 条，落在哪一端 ===")
print(z["end"].value_counts().to_string())

print(f"\n=== Q2 相应外显子长度分布（d==0 组）===")
print(z["exon_len"].describe().to_string())
print("\n外显子长度 <=3 的条数：", int((z["exon_len"] <= 3).sum()))

print(f"\n=== Q1b d==0 组的标签分布 ===")
print(z["label"].value_counts().to_string())

print(f"\n=== Q3 是否集中在少数基因 ===")
vc = z["gene"].value_counts()
print(f"涉及基因数 {len(vc)} / 总基因数 {d['gene'].nunique()}")
print(vc.head(10).to_string())

print(f"\n=== Q3b 按基因的 d==0 条数分布 ===")
print(vc.describe().to_string())

print(f"\n=== Q4 对照：d<=3 时标签分布 ===")
print(d[d["d"] <= 3]["label"].value_counts().to_string())

print(f"\n=== Q5 抽查 8 条 d==0 记录 ===")
print(z.head(8).to_string())

print(f"\n=== Q6 外显子长度总体分布（全部匹配上的）===")
print(d["exon_len"].describe().to_string())

json.dump({
    "n_d0": int(len(z)),
    "d0_by_end": {str(k): int(v) for k, v in z["end"].value_counts().items()},
    "d0_by_label": {str(k): int(v) for k, v in z["label"].value_counts().items()},
    "d0_exon_len_median": float(z["exon_len"].median()),
    "d0_exon_len_le3": int((z["exon_len"] <= 3).sum()),
    "d0_genes": int(z["gene"].nunique()),
    "all_genes": int(d["gene"].nunique()),
    "d0_per_gene_max": int(z["gene"].value_counts().max()),
}, open(f"{OUT}/s4_diag_v1.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print(f"\n[out] {OUT}/s4_diag_v1.json")
