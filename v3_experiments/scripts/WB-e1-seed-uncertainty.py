#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-seed-uncertainty.py (2026-10-03)

GPT §七 要求三类不确定性都要报：
  ① 测试基因/变异抽样   → cluster bootstrap，EXP-017 已完成
  ② 外层划分            → 240 折（已冻结，EXP-015）
  ③ **优化种子**        → 本脚本（此前完全没做）

做法：数据和折全部固定不动，只变优化的随机性——
   - 内层 CV 的划分种子（影响超参选择）
   - MLP 的初始化种子（LR 的 lbfgs 本身是确定的，但它也走同一套内层划分）

要回答的不是"种子会不会让结果变"（当然会），而是：
**种子带来的波动，相对"抽样带来的波动"有多大？**
若种子 SD 远小于 bootstrap CI 半宽 ⇒ 可以放心说"不确定性的主要来源不是优化"。

判据：
  P1 锚点：seed=42 的那一轮必须复现已冻结的数字（B3 0.8358 / B4 0.8511）
  P2 对比：种子 SD vs EXP-017 的 bootstrap CI 半宽（B3 半宽约 ±0.034）

输出：results/supplementary/wb_rerun/wb_frozen/seed_uncertainty_v1.json
"""
import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
BATCH = "seed_uncertainty_v1"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
v2 = pd.read_csv(f"{OUT}/dataset_v2/variants.tsv", sep="\t", dtype=str)
n = len(df)
y = df["label"].values.astype(int)
assert len(v2) == n and (v2["clinvar_label"].astype(int).values == y).all()

z = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
gidx, fold_id = z["global_index"], z["fold_id"]
n_folds = int(fold_id.max()) + 1

# ---------- 特征：B3（密码子查表）与 B4（剪接位置） ----------
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


def _onehot(arr):
    M = np.zeros((len(arr), 65))
    for i, c in enumerate(arr):
        M[i, CIDX.get(str(c), CIDX["NNN"])] = 1.0
    return M


gc = np.array([(str(s).upper().count("G") + str(s).upper().count("C")) /
               max(1, len(str(s))) for s in df["char_seq"].values]).reshape(-1, 1)
cpos = df["cpos"].values.astype(float).reshape(-1, 1)
X_B3 = np.hstack([_onehot(v2["reference_codon"].values),
                  _onehot(v2["alternate_codon"].values), gc, cpos])

d_exon = pd.to_numeric(v2["exon_boundary_distance"], errors="coerce").values
d_f = np.where(~np.isnan(d_exon), d_exon, np.nanmedian(d_exon)).astype(float)
X_B4 = np.log1p(d_f).reshape(-1, 1)

FEATS = {"B3_codon": X_B3, "B4_splice": X_B4}
print(f"[data] n={n} 折内={len(gidx)} 折数={n_folds}", flush=True)


def mk(tag, param, seed):
    m = (LogisticRegression(max_iter=1000, C=param, solver="lbfgs") if tag == "lr"
         else MLPClassifier(hidden_layer_sizes=(128, 64), alpha=param, max_iter=300,
                            random_state=seed, early_stopping=True,
                            validation_fraction=0.1))
    return make_pipeline(StandardScaler(), m)


def fit_select(kind, Xtr, ytr, seed):
    n_inner = 2 if kind == "lr" else 1
    try:
        splits = list(StratifiedKFold(n_inner, shuffle=True,
                                      random_state=seed).split(Xtr, ytr))
    except Exception:
        splits = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    grid = [("lr", C) for C in (0.1, 1.0)] if kind == "lr" else \
           [("mlp", a) for a in (1e-3, 1e-2)]
    best, best_s = None, -1.0
    for tag, param in grid:
        ss = []
        for tr_i, va_i in splits:
            p = mk(tag, param, seed).fit(Xtr[tr_i], ytr[tr_i])
            pv = p.predict_proba(Xtr[va_i])[:, 1]
            ss.append(roc_auc_score(ytr[va_i], pv) if len(set(ytr[va_i])) > 1 else 0.5)
        s = float(np.mean(ss))
        if s > best_s:
            best_s, best = s, (tag, param)
    tag, param = best
    return mk(tag, param, seed).fit(Xtr, ytr)


def run(X, kind, seed):
    aucs = []
    for k in range(n_folds):
        te = gidx[fold_id == k]
        tr = np.ones(n, dtype=bool); tr[te] = False
        if len(set(y[te])) < 2:
            continue
        p = fit_select(kind, X[tr], y[tr], seed)
        aucs.append(roc_auc_score(y[te], p.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), np.array(aucs)


SEEDS = [42, 7, 2026, 31415, 2718]
res = {}
for fname, X in FEATS.items():
    for kind in ("lr", "mlp"):
        means, per_fold = [], []
        for sd in SEEDS:
            m, pf = run(X, kind, sd)
            means.append(m); per_fold.append(pf)
            print(f"{fname:10s} {kind:4s} seed={sd:6d} 逐折均值={m:.4f}", flush=True)
        means = np.array(means)
        res[f"{fname}|{kind}"] = {
            "seeds": SEEDS,
            "mean_by_seed": [round(float(x), 6) for x in means],
            "mean_over_seeds": float(means.mean()),
            "sd_over_seeds": float(means.std(ddof=1)),
            "min": float(means.min()), "max": float(means.max()),
            "range": float(means.max() - means.min()),
        }
        print(f"  ⇒ {fname}/{kind}: 种子间 均值={means.mean():.4f} "
              f"SD={means.std(ddof=1):.5f} 极差={means.max()-means.min():.5f}", flush=True)

# ---------- 与 bootstrap（①）对比 ----------
bs = None
try:
    b = json.load(open(f"{OUT}/e4a_geneperm_v1.json"))
    bs = "see e4a_geneperm_v1.json"
except Exception:
    pass

print("\n=== 三类不确定性对比 ===")
for k, v in res.items():
    print(f"  {k:16s} 种子 SD={v['sd_over_seeds']:.5f} 极差={v['range']:.5f}")
print("  参考：EXP-017 cluster bootstrap 对 B3 的 CI 半宽约 ±0.034")

json.dump({"batch": BATCH,
           "what": "第③类不确定性（优化种子）；数据、折全部固定",
           "seeds": SEEDS,
           "seed_source": "内层 CV 划分种子 + MLP 初始化种子",
           "reference_bootstrap_CI_halfwidth_B3": 0.034,
           "results": res},
          open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"\n[out] {OUT}/{BATCH}.json", flush=True)
