# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | B-5 重跑：直接 cLM-pLM 配对差值（按 WB-实验设计书 v1）

相对 CodeArts 版 `analyze_b5_fig3.py` 的三处修正：
  1. **逐折训练**（他是一次性 train/test）→ 收集 out-of-fold 预测
  2. **CI 按折（基因）聚类 bootstrap**（他的 CI 未考虑折间不独立）
  3. 探针配置由训练集内公平搜索确定（他写死）
  另：分组单位统一为 gene ID（实测本数据 tx 与 gene 1:1，等价）

为什么用 pooled OOF 而不是逐折 AUC：
  240 个有效折，每折测试样本仅 3-40 个，逐折 AUC 无意义。
  故：逐折训练 → 汇集 OOF 预测 → pooled AUC；
      不确定性用「按折重抽样」表达，重抽样单位 = 折（基因）。

输出：b5_rerun_results.json
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from joblib import Parallel, delayed

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"
OUT.mkdir(exist_ok=True)
SEED = 42

df = pd.read_parquet(SYN / "synpath_variants.parquet")
y = df["label"].values
n = len(df)

split = np.load(SYN / "split_logo_cv.npz", allow_pickle=True)
valid_tx = split["valid_fold_tx_ids"]
folds = [(t, split["tx_%s_indices" % t]) for t in valid_tx
         if "tx_%s_indices" % t in split.files]
print("样本 %d | 有效折 %d | OOF 覆盖 %d" % (
    n, len(folds), sum(len(i) for _, i in folds)))

EMB = {
    "codonbert":    "codonbert_synonymous_emb.npy",
    "codonbert_hf": "codonbert_hf_synonymous_emb.npy",
    "encodon-80m":  "encodon-80m_synonymous_emb.npy",
    "mrnabert":     "mrnabert_task3_synonymous_emb.npy",
    "ESM-2-650M":   "ESM-2-650M_task3_synonymous_emb.npy",
    "ESM-1b-650M":  "ESM-1b-650M_task3_synonymous_emb.npy",
}
CLM = ["codonbert", "codonbert_hf", "encodon-80m", "mrnabert"]
PLM = ["ESM-2-650M", "ESM-1b-650M"]

# ---- 探针配置：在训练集内做一次公平搜索（小网格，兼顾速度）----
LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))


def pick_lr_config(X, yy, seed=SEED):
    best, bs = {"C": 1.0}, -1
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for cfg in LR_GRID:
        s = []
        for tr, va in skf.split(X, yy):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=1000, random_state=seed, **cfg)
            m.fit(sc.transform(X[tr]), yy[tr])
            s.append(roc_auc_score(yy[va], m.predict_proba(sc.transform(X[va]))[:, 1]))
        if s and np.mean(s) > bs:
            bs, best = np.mean(s), cfg
    return best


def fit_fold(X, tidx, cfg):
    """留出 tidx，训练后预测 tidx。返回 (indices, proba)"""
    te = np.zeros(len(y), bool)
    te[tidx] = True
    tr = ~te
    Xtr, ytr = X[tr], y[tr]
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000, random_state=SEED, **cfg)
    m.fit(sc.transform(Xtr), ytr)
    return tidx, m.predict_proba(sc.transform(X[te]))[:, 1]


results = {"_design": "WB-实验设计书 v1", "_seed": SEED,
           "_note": "逐折 LOGO 训练 + pooled OOF + 按折(基因)聚类 bootstrap CI",
           "_unit": {"n": n, "n_valid_folds": len(folds)},
           "models": {}, "paired": {}}

oof = {}
for mname, fn in EMB.items():
    p = SUPP / fn
    if not p.exists():
        print("  跳过（缺嵌入）:", mname)
        continue
    X = np.load(p)
    if X.shape[0] != n:
        print("  跳过（行数不符）:", mname)
        continue
    print("  运行", mname, X.shape)
    CFG = pick_lr_config(X, y)
    res = Parallel(n_jobs=8, verbose=0)(
        delayed(fit_fold)(X, tidx, CFG) for _, tidx in folds)
    idx = np.concatenate([r[0] for r in res])
    pr = np.concatenate([r[1] for r in res])
    order = np.argsort(idx)
    idx, pr = idx[order], pr[order]
    oof[mname] = (idx, pr)
    results["models"][mname] = {
        "n_oof": int(len(idx)),
        "auc_pooled": float(roc_auc_score(y[idx], pr)),
        "lr_config": str(CFG)}
    print("     OOF=%d  pooled AUC=%.4f  cfg=%s" % (
        len(idx), results["models"][mname]["auc_pooled"], CFG))

# ---- 配对差值 + 按折 bootstrap ----
fold_ids = {}
for k, (t, tidx) in enumerate(folds):
    for i in tidx:
        fold_ids[int(i)] = k


def boot_diff(a, b, n_boot=1000, seed=SEED):
    """按折重抽样，返回差值均值与 95% CI"""
    ia, pa = oof[a]
    ib, pb = oof[b]
    assert np.array_equal(ia, ib)
    fids = np.array([fold_ids[int(i)] for i in ia])
    uf = np.unique(fids)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(uf, size=len(uf), replace=True)
        mask = np.concatenate([np.where(fids == f)[0] for f in samp])
        try:
            da = roc_auc_score(y[ia][mask], pa[mask])
            db = roc_auc_score(y[ib][mask], pb[mask])
            diffs.append(da - db)
        except ValueError:
            continue
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), \
        float(np.percentile(diffs, 97.5))


for c in CLM:
    for pl in PLM:
        if c not in oof or pl not in oof:
            continue
        ia, pa = oof[c]
        ib, pb = oof[pl]
        pt = float(roc_auc_score(y[ia], pa) - roc_auc_score(y[ib], pb))
        m, lo, hi = boot_diff(c, pl)
        results["paired"]["%s - %s" % (c, pl)] = {
            "point_diff": pt, "boot_mean": m,
            "ci95_lo": lo, "ci95_hi": hi,
            "ci_includes_zero": bool(lo <= 0 <= hi)}
        print("  %-14s - %-12s  Δ=%+.4f  CI[%+.4f, %+.4f]  %s" % (
            c, pl, pt, lo, hi, "含0" if lo <= 0 <= hi else "★不含0"))

with open(OUT / "b5_rerun_results.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("\n已写出:", OUT / "b5_rerun_results.json")
