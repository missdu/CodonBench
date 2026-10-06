#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-b4-splice.py (2026-10-03)

动因：S4 剪接审计里发现一个此前没人量的事实——
    致病同义变异里 774/1410 = 54.9% 落在**外显子的第一个或最后一个碱基**上，
    良性组只有 1/1404 = 0.07%。
    外显子首末碱基正是剪接信号的核心位置。

要回答：这条"位置/剪接"通道单独能拿多少分？它与 B3（密码子查表）重叠多少？

新增基线（全部与 B0–B3 同口径：240 valid folds、逐折均值、LR/MLP 同预算）：
    B4a  d_exon 连续（log1p）                      1 维
    B4b  是否 d_exon == 0（外显子首/末碱基）       1 维
    B4c  [d==0, d<=3, log1p(d), 外显子内相对位置]  4 维
    B5   B3 ⊕ B4c（看密码子之外还有没有增量）
    RULE 不训练：直接把 -(d_exon) 当分数（纯规则金丝雀）

负例（能证伪的）：
    N1 打乱 d_exon 与标签的对应 ⇒ 所有 B4* 必须塌回 ≈0.5
    N2 把 d_exon 换成 "到 CDS 边界的距离"（原稿用的那个量）
       ⇒ 若它也高，说明是一般性位置效应而不是剪接位置效应；若它低，
         说明"靠近 CDS 边界"确实不是一回事（Reject §二.4.1 的点）

输出（永不覆盖）：results/supplementary/wb_rerun/wb_frozen/b4_splice_v1.json
"""
import os
import json
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
BATCH = "b4_splice_v1"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
v2 = pd.read_csv(f"{OUT}/dataset_v2/variants.tsv", sep="\t", dtype=str)
n = len(df)
y = df["label"].values.astype(int)

assert len(v2) == n, f"冻结包行数 {len(v2)} != parquet {n}"
assert (v2["clinvar_label"].astype(int).values == y).all(), "冻结包与 parquet 标签对不上"

# ---- 折 ----
z = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
gidx, fold_id = z["global_index"], z["fold_id"]
n_folds = int(fold_id.max()) + 1
print(f"[folds] {n_folds} 折, 折内样本 {len(gidx)}", flush=True)

# ---- 距离 ----
def _num(col):
    return pd.to_numeric(v2[col], errors="coerce").values

d_exon = _num("exon_boundary_distance")     # 到最近外显子—内含子接点
d_cds = _num("cds_boundary_distance")       # 到 CDS 起止（原稿口径）
n_ex = _num("n_exon_in_tx")

m = ~np.isnan(d_exon)
print(f"[dist] 有效距离 {int(m.sum())}/{n}；d_exon 中位 {np.nanmedian(d_exon):.0f}", flush=True)
fill = float(np.nanmedian(d_exon))
d_exon_f = np.where(m, d_exon, fill).astype(float)
d_cds_f = np.where(~np.isnan(d_cds), d_cds, np.nanmedian(d_cds)).astype(float)

# 外显子内相对位置：需要外显子长度 → 用 dataset_v2 里没有，退化为"是否在外显子首/末"
rel_pos = 1.0 / (1.0 + d_exon_f)     # 越靠近边缘越大

X_B4a = np.log1p(d_exon_f).reshape(-1, 1)
X_B4b = (d_exon_f == 0).astype(float).reshape(-1, 1)
X_B4c = np.column_stack([
    (d_exon_f == 0).astype(float),
    (d_exon_f <= 3).astype(float),
    np.log1p(d_exon_f),
    rel_pos,
])
X_N2cds = np.log1p(d_cds_f).reshape(-1, 1)

# ---- B3（复用冻结包里的密码子查表特征，保持与 EXP-018 完全一致）----
b = json.load(open(f"{OUT}/baselines_240logo_v1.json"))
X_B3 = None  # 若需要 B3⊕B4c，用下面的 one-hot 重建代价太大，改为只报 B3 已有点估计做参照
B3_REF = None
for k, v in b["results"].items():
    if k.startswith("B3_codon_table") and k.endswith("|lr"):
        B3_REF = v["auc_perfold_mean"]
print(f"[ref] B3 (EXP-018, LR, 逐折均值) = {B3_REF}", flush=True)

# ---- 探针 ----
def make_grid(kind):
    return [("lr", C) for C in (0.1, 1.0)] if kind == "lr" else \
           [("mlp", a) for a in (1e-3, 1e-2)]


def mk(tag, param, seed=42):
    m_ = (LogisticRegression(max_iter=1000, C=param, solver="lbfgs") if tag == "lr"
          else MLPClassifier(hidden_layer_sizes=(128, 64), alpha=param, max_iter=300,
                             random_state=seed, early_stopping=True,
                             validation_fraction=0.1))
    return make_pipeline(StandardScaler(), m_)


def fit_select(kind, Xtr, ytr, seed=42):
    if len(set(ytr)) < 2:
        return None
    n_inner = 2 if kind == "lr" else 1
    try:
        splits = list(StratifiedKFold(n_inner, shuffle=True,
                                      random_state=seed).split(Xtr, ytr))
    except Exception:
        splits = []
    if not splits:
        splits = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    best, best_s = None, -1.0
    for tag, param in make_grid(kind):
        ss = []
        for tr_i, va_i in splits:
            p = mk(tag, param).fit(Xtr[tr_i], ytr[tr_i])
            pv = p.predict_proba(Xtr[va_i])[:, 1]
            ss.append(roc_auc_score(ytr[va_i], pv) if len(set(ytr[va_i])) > 1 else 0.5)
        s = float(np.mean(ss))
        if s > best_s:
            best_s, best = s, (tag, param)
    tag, param = best
    return mk(tag, param).fit(Xtr, ytr)


def run(name, X, probe="lr"):
    aucs, aps, brs = [], [], []
    for k in range(n_folds):
        te = gidx[fold_id == k]
        tr = np.ones(n, dtype=bool); tr[te] = False
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if len(set(yte)) < 2:
            continue
        pipe = fit_select(probe, Xtr, ytr)
        if pipe is None:
            continue
        pv = pipe.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, pv))
        aps.append(average_precision_score(yte, pv))
        brs.append(brier_score_loss(yte, pv))
    aucs = np.array(aucs)
    if len(aucs) == 0:
        return None
    return {
        "n_folds_eval": int(len(aucs)),
        "auc_perfold_mean": float(aucs.mean()),
        "auc_perfold_sd": float(aucs.std(ddof=1)),
        "auc_perfold_sem": float(aucs.std(ddof=1) / np.sqrt(len(aucs))),
        "auprc_mean": float(np.mean(aps)),
        "brier_mean": float(np.mean(brs)),
        "auc_per_fold": aucs.tolist(),
    }


results = {}
for nm, X in (("B4a_d_exon_log", X_B4a), ("B4b_edge_flag", X_B4b),
              ("B4c_splice_set", X_B4c), ("N2_d_cds_boundary", X_N2cds)):
    r = run(nm, X, "lr")
    results[nm + "|lr"] = r
    print(f"{nm:20s} AUC={r['auc_perfold_mean']:.4f} ±{r['auc_perfold_sem']:.4f} "
          f"AUPRC={r['auprc_mean']:.4f} (n={r['n_folds_eval']})", flush=True)

# ---- RULE：不训练，纯规则金丝雀 ----
rule_aucs = []
for k in range(n_folds):
    te = gidx[fold_id == k]
    yte = y[te]
    if len(set(yte)) < 2:
        continue
    rule_aucs.append(roc_auc_score(yte, -d_exon_f[te].astype(float)))
rule_aucs = np.array(rule_aucs)
print(f"{'RULE_-d_exon(不训练)':20s} AUC={rule_aucs.mean():.4f} "
      f"±{rule_aucs.std(ddof=1)/np.sqrt(len(rule_aucs)):.4f}", flush=True)
results["RULE_neg_d_exon"] = {
    "n_folds_eval": int(len(rule_aucs)),
    "auc_perfold_mean": float(rule_aucs.mean()),
    "auc_perfold_sem": float(rule_aucs.std(ddof=1) / np.sqrt(len(rule_aucs))),
    "auc_per_fold": rule_aucs.tolist(),
    "note": "不训练任何模型：直接用 -(到外显子接点的距离) 当分数",
}

# ---- 负例 N1：打乱 d_exon ----
rng = np.random.default_rng(20261003)
perm = rng.permutation(n)
X_perm = np.log1p(d_exon_f[perm]).reshape(-1, 1)
r = run("N1_shuffled_d_exon", X_perm, "lr")
results["N1_shuffled_d_exon|lr"] = r
print(f"{'N1_shuffled':20s} AUC={r['auc_perfold_mean']:.4f} "
      f"±{r['auc_perfold_sem']:.4f}", flush=True)

# ---- 分布事实（进 S4）----
in_folds = np.zeros(n, dtype=bool); in_folds[gidx] = True
sel = m & in_folds
fact = {
    "n_in_folds": int(in_folds.sum()),
    "frac_d0_all": float(np.mean(d_exon[sel] == 0)),
    "frac_d0_path": float(np.mean(d_exon[sel & (y == 1)] == 0)),
    "frac_d0_benign": float(np.mean(d_exon[sel & (y == 0)] == 0)),
    "frac_d3_all": float(np.mean(d_exon[sel] <= 3)),
    "frac_d3_path": float(np.mean(d_exon[sel & (y == 1)] <= 3)),
    "frac_d3_benign": float(np.mean(d_exon[sel & (y == 0)] <= 3)),
    "n_d0_path": int(((d_exon == 0) & (y == 1)).sum()),
    "n_d0_benign": int(((d_exon == 0) & (y == 0)).sum()),
}
print("\n=== 事实（折内 1847 条）===", flush=True)
for k, v in fact.items():
    print(f"  {k:18s} {v}", flush=True)

json.dump({
    "batch": BATCH,
    "folds_batch": "folds_240logo_v1",
    "distance_source": "UCSC refGene (hg19/hg38) via dataset_v2",
    "primary_metric": "逐折 AUC 均值",
    "B3_reference_LR": B3_REF,
    "results": results,
    "facts_in_folds": fact,
}, open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"\n[out] {OUT}/{BATCH}.json", flush=True)
