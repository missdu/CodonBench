#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-b4-vs-b3.py (2026-10-03)

上一轮（WB-e1-b4-splice.py）的结果：
    「到外显子接点的距离」这一维，不训练直接当分数 ⇒ 逐折 AUC = 0.8511
    而密码子查表 B3 = 0.8358
    ⇒ 一个**不读序列**的量，压过了读序列的密码子特征。

于是必须回答：B3 的高分，有多少其实来自剪接位置？

做法（在两个互斥子集上分别重估，同一批 240 折、同一探针预算）：
    近端子集  d_exon <= 3   （剪接信号核心区）
    远端子集  d_exon >  3   （远离剪接位点）
若 B3 在远端子集上大幅掉落 ⇒ B3 的分数主要由"靠近剪接位点的那部分样本"撑起，
则"模型学到了密码子规律"的解释不成立。

另外做组合与残差：
    B3 ⊕ B4a          看剪接位置之外 B3 还有没有增量
    锚点校验          重建的 B3 必须复现 0.8358（差 >1e-3 即判构造不一致）

输出：results/supplementary/wb_rerun/wb_frozen/b4_vs_b3_v1.json
"""
import os
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
BATCH = "b4_vs_b3_v1"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
v2 = pd.read_csv(f"{OUT}/dataset_v2/variants.tsv", sep="\t", dtype=str)
n = len(df)
y = df["label"].values.astype(int)
assert len(v2) == n and (v2["clinvar_label"].astype(int).values == y).all()

z = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
gidx, fold_id = z["global_index"], z["fold_id"]
n_folds = int(fold_id.max()) + 1

# ---------- B3：密码子查表（与 EXP-018 配方一致） ----------
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON2AA = {}
_i = 0
for _x in _b:
    for _y in _b:
        for _z in _b:
            CODON2AA[_x + _y + _z] = _aa[_i]; _i += 1
CIDX = {a + b + c: i for i, (a, b, c) in enumerate(
    (x, y_, z_) for x in "ACGT" for y_ in "ACGT" for z_ in "ACGT")}
CIDX["NNN"] = len(CIDX)

ref_codon = v2["reference_codon"].values
alt_codon = v2["alternate_codon"].values


def _onehot(arr):
    M = np.zeros((len(arr), 65))
    for i, c in enumerate(arr):
        M[i, CIDX.get(str(c), CIDX["NNN"])] = 1.0
    return M


gc = np.array([(str(s).upper().count("G") + str(s).upper().count("C")) /
               max(1, len(str(s))) for s in df["char_seq"].values]).reshape(-1, 1)
cpos = df["cpos"].values.astype(float).reshape(-1, 1)
X_B3 = np.hstack([_onehot(ref_codon), _onehot(alt_codon), gc, cpos])

# ---------- B4a：剪接位置 ----------
d_exon = pd.to_numeric(v2["exon_boundary_distance"], errors="coerce").values
m_valid = ~np.isnan(d_exon)
fill = float(np.nanmedian(d_exon))
d_f = np.where(m_valid, d_exon, fill).astype(float)
X_B4 = np.log1p(d_f).reshape(-1, 1)
X_B3B4 = np.hstack([X_B3, X_B4])

print(f"[data] n={n}, 折内={len(gidx)}, B3 维={X_B3.shape[1]}, 有效距离={int(m_valid.sum())}",
      flush=True)


def fit_lr(Xtr, ytr, seed=42):
    best, best_s, best_C = None, -1.0, 1.0
    try:
        splits = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(Xtr, ytr))
    except Exception:
        splits = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    for C in (0.1, 1.0):
        ss = []
        for tr_i, va_i in splits:
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, C=C, solver="lbfgs"))
            p.fit(Xtr[tr_i], ytr[tr_i])
            pv = p.predict_proba(Xtr[va_i])[:, 1]
            ss.append(roc_auc_score(ytr[va_i], pv) if len(set(ytr[va_i])) > 1 else 0.5)
        s = float(np.mean(ss))
        if s > best_s:
            best_s, best_C = s, C
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, C=best_C, solver="lbfgs"))
    p.fit(Xtr, ytr)
    return p


def run(X, subset_mask=None, tag=""):
    """subset_mask：None = 全部；否则只在 mask 命中的**测试**样本上评估。"""
    aucs, aps, sizes = [], [], []
    for k in range(n_folds):
        te_all = gidx[fold_id == k]
        te = te_all[subset_mask[te_all]] if subset_mask is not None else te_all
        if len(te) == 0 or len(set(y[te])) < 2:
            continue
        tr = np.ones(n, dtype=bool); tr[te_all] = False
        if subset_mask is not None:          # 训练也限制在同种子集，避免分布混杂
            tr = tr & subset_mask
        if len(set(y[tr])) < 2:
            continue
        p = fit_lr(X[tr], y[tr])
        pv = p.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], pv))
        aps.append(average_precision_score(y[te], pv))
        sizes.append(len(te))
    if not aucs:
        return None
    aucs = np.array(aucs)
    return {"n_folds_eval": int(len(aucs)),
            "auc_mean": float(aucs.mean()),
            "auc_sem": float(aucs.std(ddof=1) / np.sqrt(len(aucs))),
            "auprc_mean": float(np.mean(aps)),
            "median_fold_n": float(np.median(sizes)),
            "auc_per_fold": aucs.tolist()}


near = m_valid & (d_exon <= 3)
far = m_valid & (d_exon > 3)
print(f"[split] 近端(d<=3)={int(near.sum())}  远端(d>3)={int(far.sum())}", flush=True)

res = {}
for nm, X in (("B3_codon", X_B3), ("B4_splice", X_B4), ("B3+B4", X_B3B4)):
    for sname, sm in (("all", None), ("near_d<=3", near), ("far_d>3", far)):
        r = run(X, sm)
        key = f"{nm}|{sname}"
        res[key] = r
        if r is None:
            print(f"{key:24s} 无可用折", flush=True)
        else:
            print(f"{key:24s} AUC={r['auc_mean']:.4f} ±{r['auc_sem']:.4f} "
                  f"(n={r['n_folds_eval']}折, 中位样本{r['median_fold_n']:.0f})", flush=True)

# ---------- pooled 辅助口径（逐折样本太少，pooled 更稳；只作对照不作主结果） ----------
def pooled(X, subset_mask, n_rep=1):
    """在 subset 上按折做 out-of-fold 预测，汇总成一个 AUC。
    注意：这是辅助口径，主口径仍是逐折均值（GPT §七.3）。"""
    te_all_idx = gidx
    if subset_mask is not None:
        te_all_idx = gidx[subset_mask[gidx]]
    if len(te_all_idx) == 0:
        return None
    yte_all, score = [], []
    for k in range(n_folds):
        te_all = gidx[fold_id == k]
        te = te_all[subset_mask[te_all]] if subset_mask is not None else te_all
        if len(te) == 0 or len(set(y[te])) < 2:
            continue
        tr = np.ones(n, dtype=bool); tr[te_all] = False
        if subset_mask is not None:
            tr = tr & subset_mask
        if len(set(y[tr])) < 2:
            continue
        p = fit_lr(X[tr], y[tr])
        pv = p.predict_proba(X[te])[:, 1]
        yte_all.extend(y[te].tolist()); score.extend(pv.tolist())
    if len(set(yte_all)) < 2:
        return None
    return {"auc": float(roc_auc_score(yte_all, score)), "n": len(yte_all)}


pooled_res = {}
for nm, X in (("B3_codon", X_B3), ("B4_splice", X_B4), ("B3+B4", X_B3B4)):
    for sname, sm in (("all", None), ("near_d<=3", near), ("far_d>3", far)):
        pooled_res[f"{nm}|{sname}"] = pooled(X, sm)
        r = pooled_res[f"{nm}|{sname}"]
        print(f"[pooled] {nm:9s} {sname:9s} AUC={r['auc']:.4f} (n={r['n']})"
              if r else f"[pooled] {nm} {sname} NA", flush=True)

# ---------- 锚点校验 ----------
ref = json.load(open(f"{OUT}/baselines_240logo_v1.json"))
b3_ref = None
for k, v in ref["results"].items():
    if k.startswith("B3_codon_table") and k.endswith("|lr"):
        b3_ref = v["auc_perfold_mean"]
mine = res["B3_codon|all"]["auc_mean"] if res.get("B3_codon|all") else None
diff = abs(b3_ref - mine) if mine is not None else None
print(f"\n[锚点] 官方 B3 = {b3_ref:.6f}；本次重建 = "
      f"{('%.6f' % mine) if mine is not None else 'NA'}；差 = "
      f"{('%.6f' % diff) if diff is not None else 'NA'}", flush=True)

json.dump({
    "batch": BATCH,
    "anchor": {"b3_official": b3_ref, "b3_rebuilt": mine, "abs_diff": diff,
               "verdict": "PASS" if diff is not None and diff < 1e-3 else "CHECK"},
    "primary_per_fold": res,
    "auxiliary_pooled": pooled_res,
    "subset_sizes": {"near_d<=3": int(near.sum()), "far_d>3": int(far.sum()),
                     "in_folds": int(len(gidx))},
}, open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"\n[out] {OUT}/{BATCH}.json", flush=True)
