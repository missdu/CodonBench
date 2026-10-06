#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-cohort-declare.py (2026-10-03)

Reject §二.3.2：排除掉的那批转录本折会**改变评价人群**，必须量化后声明。
要写进稿子的是两件事：
  1. 谁进了 240 折、谁没进，两群人在标签率和 ClinVar 证据等级上差多少
  2. 每折只有中位 5 个变异 ⇒ 逐折 AUC 噪声大，pooled 只作辅助（GPT §七.3）

顺便给出 §二.5.3 要的 pooled vs 逐折 的口径对照数字。

输出：results/supplementary/wb_rerun/wb_frozen/cohort_declaration_v1.json
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
n = len(df)
y = df["label"].values.astype(int)
assert len(v2) == n and (v2["clinvar_label"].astype(int).values == y).all()

z = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
gidx, fold_id = z["global_index"], z["fold_id"]
fold_tx, fold_n, fold_pos, fold_neg = (z["fold_tx_id"], z["fold_n"],
                                       z["fold_pos"], z["fold_neg"])

in_folds = np.zeros(n, dtype=bool)
in_folds[gidx] = True
out_folds = ~in_folds

rev = df["ReviewStatus"].values.astype(str)
gene = df["gene_symbol"].values


def _prof(mask, tag):
    d = {
        "n": int(mask.sum()),
        "pos_rate": round(float(y[mask].mean()), 4) if mask.any() else None,
        "n_genes": int(len(set(gene[mask]))),
        "review_status": {str(k): int(v_) for k, v_ in
                          pd.Series(rev[mask]).value_counts().items()},
    }
    print(f"--- {tag} ---")
    print(f"    n={d['n']}  标签率={d['pos_rate']}  基因数={d['n_genes']}")
    for k, v_ in list(d["review_status"].items())[:6]:
        print(f"      {k}: {v_}")
    return d


print("=== 评价人群对比 ===", flush=True)
allp = _prof(np.ones(n, dtype=bool), "全集 2840")
inp = _prof(in_folds, "折内 1847")
outp = _prof(out_folds, "未入折 993")

# 转录本未入折的原因：每转录本变异数不足
per_tx = pd.Series(df["tx_id"].values).value_counts()
tx_in = set(str(t) for t in fold_tx)
tx_out = [t for t in per_tx.index if str(t) not in tx_in]
print(f"\n=== 折的构成 ===")
print(f"  入折转录本数 {len(tx_in)}；未入折 {len(tx_out)}")
print(f"  入折转录本的变异数：中位 {np.median(fold_n):.0f} "
      f"最小 {fold_n.min()} 最大 {fold_n.max()}")
print(f"  未入折转录本的变异数：中位 {per_tx.loc[tx_out].median():.0f} "
      f"最小 {per_tx.loc[tx_out].min()} 最大 {per_tx.loc[tx_out].max()}")

# 入折的判据是什么？（从分布反推）
print(f"  未入折转录本里变异数 >=3 的个数："
      f"{int((per_tx.loc[tx_out] >= 3).sum())}")
print(f"  未入折转录本里变异数 >=5 的个数："
      f"{int((per_tx.loc[tx_out] >= 5).sum())}")

# 每折正负样本是否都 >=1（否则该折算不出 AUC）
both = (fold_pos >= 1) & (fold_neg >= 1)
print(f"\n=== 每折可评估性 ===")
print(f"  240 折中正负样本都>=1 的：{int(both.sum())}")
print(f"  折内样本数分位：{np.percentile(fold_n, [10,25,50,75,90]).tolist()}")

# 口径对照：pooled vs 逐折（用已冻结的 B3 结果做例子）
b3 = None
try:
    b = json.load(open(f"{OUT}/baselines_240logo_v1.json"))
    for k, v_ in b["results"].items():
        if k.startswith("B3_codon_table") and k.endswith("|lr"):
            b3 = v_
except Exception:
    pass

decl = {
    "cohort": {"all_2840": allp, "in_folds_1847": inp, "out_folds_993": outp},
    "fold_construction": {
        "n_folds": int(len(fold_tx)),
        "n_tx_in_folds": int(len(tx_in)),
        "n_tx_out_folds": int(len(tx_out)),
        "tx_variant_count_in_folds": {"median": float(np.median(fold_n)),
                                      "min": int(fold_n.min()),
                                      "max": int(fold_n.max())},
        "tx_variant_count_out_folds": {"median": float(per_tx.loc[tx_out].median()),
                                       "min": int(per_tx.loc[tx_out].min()),
                                       "max": int(per_tx.loc[tx_out].max())},
        "folds_with_both_classes": int(both.sum()),
        "fold_size_percentiles": {str(q): float(v_) for q, v_ in
                                  zip([10, 25, 50, 75, 90],
                                      np.percentile(fold_n, [10, 25, 50, 75, 90]))},
    },
    "train_set_rule": "训练集 = 全集(2840) 减去该折测试集 ⇒ 含未进入任何折的 993 条变异",
    "primary_vs_auxiliary": {
        "primary": "逐折 AUC 均值（每折 = 一个留出基因；折内 AUC 即基因内 AUC）",
        "auxiliary": "pooled out-of-fold，仅作对照",
        "B3_example": {"per_fold_mean": b3["auc_perfold_mean"] if b3 else None,
                       "per_fold_sem": b3["auc_perfold_sem"] if b3 else None,
                       "per_fold_sd": b3["auc_perfold_sd"] if b3 else None},
    },
}
json.dump(decl, open(f"{OUT}/cohort_declaration_v1.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\n[out] {OUT}/cohort_declaration_v1.json", flush=True)
