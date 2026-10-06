# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | G3（未见同源簇）—— 真正做出来

【为什么之前做不了】
  CodeArts 版用"基因符号首字母"当同源代理 —— 首字母与同源性无关，是无效定义。
  本地无同源数据库，Ensembl REST / NCBI E-utilities 实测均不通。
  本版用 **HGNC gene_group**（HUGO 官方审编的基因家族）作为家族归属，
  按 HGNC_ID 精确匹配，498 个基因中 457 个有家族注释。

【要回答的问题】
  把测试基因的**整个同源家族**都从训练集拿掉，模型还剩下多少预测力？

【必须堵住的质疑】
  拿掉家族 = 同时拿掉了 44% 的训练基因。掉分可能只是"训练基因变少了"，
  与"同源"无关。所以加一组对照：
    G3-ctrl —— 随机剔除**同样数量**的训练基因（不看家族）
  若 G3 明显低于 G3-ctrl ⇒ 掉的是"同源支持"，不只是数据量。

【四个臂，同一测试集，训练样本量匹配】
  G1      已知基因（测试基因的其他变异也在训练集里）
  G2      未见基因（测试基因整个拿掉）
  G3      未见家族（测试基因 + 其全部家族同伴拿掉）
  G3-ctrl 未见基因 + 随机拿掉同样个数的基因（多样性对照）

【重复】5 次独立的基因划分，报均值 ± SD；G3-ctrl 每次重复内另跑 3 个随机种子
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"

# ---------- 1. 数据 ----------
df = pd.read_parquet(SYN / "synpath_variants.parquet")
y = df["label"].values
genes = df["gene_symbol"].values
n = len(df)

CLU = json.load(open(OUT / "g3_gene_clusters.json", encoding="utf-8"))
g2groups = CLU["gene_to_groups"]
all_genes = np.array(sorted(pd.unique(genes)))
print("样本 %d | 基因 %d | 有家族注释 %d"
      % (n, len(all_genes), sum(1 for g in all_genes if g2groups.get(g))))

# ---------- 2. 背景基线 ----------
def background_features(d):
    seqs = d["char_seq"].values
    gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
    from collections import Counter
    cnt = Counter()
    for s in seqs:
        for i in range(len(s) - 2):
            cnt[s[i:i + 3]] += 1
    top = [k for k, _ in cnt.most_common(16)]
    kmer = np.array([[s.count(k) / max(1, len(s)) for k in top] for s in seqs])
    pos = d["cpos"].values.reshape(-1, 1).astype(float)
    return np.hstack([gc.reshape(-1, 1), kmer, pos])


BG = background_features(df)

LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))
MLP_GRID = list(ParameterGrid({"hidden_layer_sizes": [(128, 64)],
                               "alpha": [1e-3, 1e-2]}))


def _mk(kind, cfg, seed):
    if kind == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed, **cfg)
    return MLPClassifier(max_iter=200, early_stopping=True, random_state=seed, **cfg)


def fair_probe(Xtr, ytr, kind, seed=42):
    grid = LR_GRID if kind == "lr" else MLP_GRID
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    best, best_score = None, -1
    for cfg in grid:
        sc = []
        for tr_i, va_i in skf.split(Xtr, ytr):
            m = _mk(kind, cfg, seed)
            s = StandardScaler().fit(Xtr[tr_i])
            m.fit(s.transform(Xtr[tr_i]), ytr[tr_i])
            p = m.predict_proba(s.transform(Xtr[va_i]))[:, 1]
            if len(np.unique(ytr[va_i])) < 2:
                continue
            sc.append(roc_auc_score(ytr[va_i], p))
        if sc and np.mean(sc) > best_score:
            best_score, best = np.mean(sc), cfg
    return best


def fit_predict(X, tr, te, kind="lr", seed=42):
    Xtr, ytr = X[tr], y[tr]
    cfg = fair_probe(Xtr, ytr, kind, seed)
    sc = StandardScaler().fit(Xtr)
    m = _mk(kind, cfg, seed)
    m.fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(roc_auc_score(y[te], p)), str(cfg)


def subsample_idx(idx, size, seed):
    idx = np.asarray(idx)
    r = np.random.RandomState(seed)
    r.shuffle(idx)
    return idx[:size]


# ---------- 3. 四个臂，5 次重复 ----------
N_REPEAT = 5
N_CTRL_SEED = 3
results = {"_design": "G3 真实同源/家族划分（HGNC gene_group）",
           "_n_repeat": N_REPEAT, "_n_ctrl_seed": N_CTRL_SEED,
           "cluster_def": "DEF-A：任一共享 HGNC 家族即视为同源",
           "per_repeat": {}, "models": {}, "baseline": {}}

EMB = {
    "codonbert":    "codonbert_synonymous_emb.npy",
    "codonbert_hf": "codonbert_hf_synonymous_emb.npy",
    "encodon-80m":  "encodon-80m_synonymous_emb.npy",
    "mrnabert":     "mrnabert_task3_synonymous_emb.npy",
    "ESM-2-650M":   "ESM-2-650M_task3_synonymous_emb.npy",
    "ESM-1b-650M":  "ESM-1b-650M_task3_synonymous_emb.npy",
}

# 预加载嵌入
Xs = {}
for mname, fn in EMB.items():
    p = SUPP / fn
    if p.exists():
        X = np.load(p)
        if X.shape[0] == n:
            Xs[mname] = X
print("可用嵌入:", list(Xs))

for rep in range(N_REPEAT):
    rng = np.random.RandomState(rep)
    sh = all_genes.copy()
    rng.shuffle(sh)
    n_test = int(len(sh) * 0.20)
    # 用 list 固定顺序：字符串 set 的迭代顺序受 PYTHONHASHSEED 随机化影响，
    # 直接遍历 set 会让划分不可复现
    test_genes = sorted(sh[:n_test])
    test_gene_set = set(test_genes)
    train_genes = [g for g in sh[n_test:]]

    te_mask = np.array([g in test_gene_set for g in genes])
    pool_mask = np.array([g in set(train_genes) for g in genes])

    # 测试基因涉及的全部家族
    tg_groups = set()
    for g in test_genes:
        tg_groups |= set(g2groups.get(g, []))
    # 训练池中需要剔除的"家族同伴"
    fam_mates = [g for g in train_genes if set(g2groups.get(g, [])) & tg_groups]

    # G1：训练池 + 每个测试基因的一半样本
    g1_extra = np.zeros(n, dtype=bool)
    r2 = np.random.RandomState(1000 + rep)
    for g in test_genes:
        idx = np.where(genes == g)[0]
        if len(idx) > 1:
            idx = idx.copy()
            r2.shuffle(idx)
            g1_extra[idx[:max(1, len(idx) // 2)]] = True
    g1_pool = pool_mask | g1_extra

    # 各臂可用的训练基因
    arm_genes = {
        "G1": None,                                   # 用布尔掩码
        "G2": set(train_genes),
        "G3": set(train_genes) - set(fam_mates),
    }
    arm_masks = {
        "G1": g1_pool,
        "G2": pool_mask,
        "G3": np.array([g in arm_genes["G3"] for g in genes]),
    }
    # G3-ctrl：随机剔除与 G3 同样个数的训练基因
    ctrl_masks = {}
    for cs in range(N_CTRL_SEED):
        rr = np.random.RandomState(2000 + 100 * rep + cs)
        tg = list(train_genes)
        rr.shuffle(tg)
        drop = set(tg[:len(fam_mates)])
        keep = set(train_genes) - drop
        ctrl_masks["G3ctrl_s%d" % cs] = np.array([g in keep for g in genes])

    all_masks = dict(arm_masks)
    all_masks.update(ctrl_masks)

    # 训练样本量匹配
    sizes = {k: int(v.sum()) for k, v in all_masks.items()}
    nmatch = min(sizes.values())
    matched = {}
    for k, v in all_masks.items():
        idx = np.where(v)[0]
        sel = subsample_idx(idx, nmatch, 3000 + rep)
        m = np.zeros(n, dtype=bool)
        m[sel] = True
        matched[k] = m

    results["per_repeat"]["rep%d" % rep] = {
        "n_test_samples": int(te_mask.sum()),
        "n_test_genes": len(test_genes),
        "n_train_genes_G2": len(train_genes),
        "n_fam_mates_removed": len(fam_mates),
        "n_train_genes_G3": len(arm_genes["G3"]),
        "n_train_matched": int(nmatch),
    }
    print("\n[rep %d] 测试基因 %d | 训练基因 G2 %d → G3 %d (剔除家族同伴 %d) | 匹配训练样本 %d"
          % (rep, len(test_genes), len(train_genes), len(arm_genes["G3"]),
             len(fam_mates), nmatch))

    # 背景基线
    for k, m in matched.items():
        a, _ = fit_predict(BG, m, te_mask, "lr")
        results["baseline"].setdefault(k, []).append(a)

    # 各模型
    for mname, X in Xs.items():
        results["models"].setdefault(mname, {})
        for k, m in matched.items():
            a, cfg = fit_predict(X, m, te_mask, "lr")
            results["models"][mname].setdefault(k, []).append(a)
        # Δ_add（仅 rep0，MLP，与 E2 口径一致）
        if rep == 0:
            Xa = np.hstack([BG, X])
            d = {}
            for k in ["G1", "G2", "G3"]:
                a_full, _ = fit_predict(Xa, matched[k], te_mask, "mlp")
                a_bg, _ = fit_predict(BG, matched[k], te_mask, "mlp")
                d[k] = float(a_full - a_bg)
            results["models"][mname]["delta_add_rep0"] = d
        print("   %s: " % mname +
              " ".join("%s %.3f" % (k, results["models"][mname][k][-1])
                       for k in ["G1", "G2", "G3"]))

# ---------- 4. 汇总 ----------
def agg(v):
    v = np.asarray(v, dtype=float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)}


summary = {"baseline": {k: agg(v) for k, v in results["baseline"].items()},
           "models": {}}
ctrl_keys = [k for k in results["baseline"] if k.startswith("G3ctrl")]
for mname, d in results["models"].items():
    summary["models"][mname] = {k: agg(v) for k, v in d.items()
                                if isinstance(v, list)}
    if "delta_add_rep0" in d:
        summary["models"][mname]["delta_add_rep0"] = d["delta_add_rep0"]
    # G3 vs G3-ctrl 对照
    g3 = np.asarray(d["G3"], dtype=float)
    # 每次重复内，先对 3 个随机剔除种子取平均 ⇒ 形状 (5,)
    ctrl_per_rep = np.array([[d[k][i] for k in ctrl_keys]
                             for i in range(N_REPEAT)]).mean(axis=1)
    diff = g3 - ctrl_per_rep
    summary["models"][mname]["G3_minus_ctrl"] = {
        "mean_diff": round(float(diff.mean()), 4),
        "sd_diff": round(float(diff.std(ddof=1)), 4),
        "per_repeat": [round(float(x), 4) for x in diff],
        "g3_mean": round(float(g3.mean()), 4),
        "ctrl_mean": round(float(ctrl_per_rep.mean()), 4),
    }
results["summary"] = summary

print("\n\n========== 汇总（5 次重复，均值 ± SD）==========")
print("背景基线 LR:")
for k in ["G1", "G2", "G3"] + ctrl_keys:
    v = summary["baseline"][k]
    print("   %-10s %.4f ± %.4f" % (k, v["mean"], v["sd"]))
print("\n各模型:")
for mname in summary["models"]:
    s = summary["models"][mname]
    print("  %s" % mname)
    for k in ["G1", "G2", "G3"] + ctrl_keys:
        print("     %-10s %.4f ± %.4f" % (k, s[k]["mean"], s[k]["sd"]))
    gm = s["G3_minus_ctrl"]
    print("     G3 − G3ctrl = %.4f ± %.4f  (逐次 %s)"
          % (gm["mean_diff"], gm["sd_diff"], gm["per_repeat"]))

with open(OUT / "g3_rerun_results.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("\n已存:", OUT / "g3_rerun_results.json")
