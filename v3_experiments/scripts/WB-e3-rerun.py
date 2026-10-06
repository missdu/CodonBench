# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E3 重做（扩容）：支持臂 vs 排除臂（按 WB-实验设计书 v1）

CodeArts 版 `run_e3_support_exclude.py` 的问题：
  * n_test = 65 —— 太小，且不清楚来自多少基因（按"有效样本量是基因数"这条，几乎不可用）
  * 未做训练规模匹配（支持臂与排除臂训练集大小不同）

本设计的修正：
  1. **逐基因留一**：每个有 >=2 个变异的基因留 1 个变异作测试 → 测试集大幅扩容
  2. **同一测试集**：支持臂与排除臂在**完全相同**的测试样本上评估
  3. **训练规模匹配**：支持臂下采样到与排除臂相同的训练量
  4. **配对 + 按基因聚类 bootstrap CI**

定义：
  支持臂 training = 其余全部样本（含该基因的其他变异）
  排除臂 training = 其余全部样本，但**去掉该基因的全部变异**

输出：e3_rerun_results.json
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
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
genes = df["gene_symbol"].values
n = len(df)

# ---- 逐基因留一 ----
rng = np.random.RandomState(SEED)
by_gene = defaultdict(list)
for i, g in enumerate(genes):
    by_gene[g].append(i)

tests = []
for g, idx in by_gene.items():
    if len(idx) < 2:
        continue                      # 只有 1 个变异的基因无法做"同基因支持"
    idx = np.array(idx)
    rng.shuffle(idx)
    tests.append((g, int(idx[0])))    # 留 1 个作测试

tests.sort(key=lambda x: x[1])
print("可用基因（>=2 变异）: %d / %d" % (len(tests), len(by_gene)))
print("测试集样本数: %d  (CodeArts 版为 65)" % len(tests))

LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))


def pick_cfg(X, yy, seed=SEED):
    best, bs = {"C": 1.0}, -1
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for cfg in LR_GRID:
        s = []
        for tr, va in skf.split(X, yy):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=1000, random_state=seed, **cfg)
            m.fit(sc.transform(X[tr]), yy[tr])
            s.append(roc_auc_score(yy[va],
                     m.predict_proba(sc.transform(X[va]))[:, 1]))
        if s and np.mean(s) > bs:
            bs, best = np.mean(s), cfg
    return best


def one_gene(X, g, tidx, cfg):
    """返回 (tidx, p_support, p_exclude, n_train_support, n_train_exclude)"""
    gi = np.array(by_gene[g])
    te = np.zeros(n, bool); te[tidx] = True

    # 支持臂：其余全部（含该基因其他变异）
    sup = ~te
    # 排除臂：其余全部，再去掉该基因全部其他变异
    exc = ~te
    exc[gi] = False

    # 训练规模匹配：支持臂下采样到排除臂大小
    ns, ne = int(sup.sum()), int(exc.sum())
    if ns > ne:
        sidx = np.where(sup)[0]
        r = np.random.RandomState(SEED + tidx)
        r.shuffle(sidx)
        sup = np.zeros(n, bool); sup[sidx[:ne]] = True

    def fit(mask):
        sc = StandardScaler().fit(X[mask])
        m = LogisticRegression(max_iter=1000, random_state=SEED, **cfg)
        m.fit(sc.transform(X[mask]), y[mask])
        return m.predict_proba(sc.transform(X[te]))[:, 1][0]

    return tidx, fit(sup), fit(exc), int(sup.sum()), int(exc.sum())


EMB = {
    "codonbert":    "codonbert_synonymous_emb.npy",
    "codonbert_hf": "codonbert_hf_synonymous_emb.npy",
    "encodon-80m":  "encodon-80m_synonymous_emb.npy",
    "mrnabert":     "mrnabert_task3_synonymous_emb.npy",
    "ESM-2-650M":   "ESM-2-650M_task3_synonymous_emb.npy",
    "ESM-1b-650M":  "ESM-1b-650M_task3_synonymous_emb.npy",
}

results = {"_design": "WB-实验设计书 v1", "_seed": SEED,
           "_note": "逐基因留一 + 支持/排除臂训练规模匹配 + 按基因聚类 bootstrap",
           "_n_test": len(tests), "_n_genes": len(tests),
           "_compare": "CodeArts 版 n_test=65", "models": {}}

for mname, fn in EMB.items():
    p = SUPP / fn
    if not p.exists():
        print("  跳过（缺嵌入）:", mname); continue
    X = np.load(p)
    if X.shape[0] != n:
        print("  跳过（行数不符）:", mname); continue
    print("  运行", mname)
    cfg = pick_cfg(X, y)
    res = Parallel(n_jobs=8, verbose=0)(
        delayed(one_gene)(X, g, t, cfg) for g, t in tests)
    tidx = np.array([r[0] for r in res])
    ps = np.array([r[1] for r in res])
    pe = np.array([r[2] for r in res])
    ntr = int(np.median([r[3] for r in res]))
    yt = y[tidx]

    a_s = float(roc_auc_score(yt, ps))
    a_e = float(roc_auc_score(yt, pe))
    delta = a_s - a_e

    # 按基因聚类 bootstrap
    r2 = np.random.RandomState(SEED)
    diffs = []
    for _ in range(1000):
        samp = r2.choice(len(tidx), size=len(tidx), replace=True)
        try:
            diffs.append(roc_auc_score(yt[samp], ps[samp])
                         - roc_auc_score(yt[samp], pe[samp]))
        except ValueError:
            continue
    diffs = np.array(diffs)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    results["models"][mname] = {
        "auc_support": a_s, "auc_exclude": a_e, "delta": delta,
        "ci95_lo": lo, "ci95_hi": hi, "ci_includes_zero": bool(lo <= 0 <= hi),
        "n_test": int(len(tidx)), "n_train_matched": ntr, "cfg": str(cfg)}
    print("     support=%.4f exclude=%.4f Δ=%+.4f CI[%+.4f,%+.4f] %s" % (
        a_s, a_e, delta, lo, hi, "含0" if lo <= 0 <= hi else "★不含0"))

with open(OUT / "e3_rerun_results.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("\n已写出:", OUT / "e3_rerun_results.json")
