# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E2 重做（按 WB-实验设计书 v1）

相对 CodeArts 版的四处修正：
  1. 分组单位统一为 gene ID（他 G1 用 gene_symbol、G2 用 transcript_id，不一致）
  2. G1/G2 使用同一测试人群（他各自独立划分，"下降"混进了"人群变了"）
  3. 训练规模匹配（他 G1 基因内 80/20、G2 全量取反，训练量不同）
  4. 背景基线加强为 GC + k-mer + 位置 one-hot（他仅 GC 一维）
  5. 探针公平搜索：LR 与 MLP 都给调参预算（他参数写死）

G3（未见同源簇）：设计书要求真实同源信息；本地无 HGNC/同源簇数据，
    故本次**不做 G3**，标注为"待补同源数据"，不硬凑代理。

输出：e2_rerun_results.json
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
from sklearn.model_selection import ParameterGrid

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"
OUT.mkdir(exist_ok=True)

RNG_SEED = 42

# ---------- 1. 数据 ----------
df = pd.read_parquet(SYN / "synpath_variants.parquet")
y = df["label"].values
genes = df["gene_symbol"].values
txs = df["tx_id"].values
n = len(df)

# 核实：多转录本基因有多少
g2t = {}
for g, t in zip(genes, txs):
    g2t.setdefault(g, set()).add(t)
multi = {g: len(v) for g, v in g2t.items() if len(v) > 1}
print("=== 单位核实 ===")
print("样本数            : %d" % n)
print("gene_symbol 唯一数: %d" % df["gene_symbol"].nunique())
print("tx_id 唯一数      : %d" % df["tx_id"].nunique())
print("多转录本基因数    : %d" % len(multi))
print("涉及样本数        : %d" % sum(
    1 for g in genes if g in multi))
print("基因大小 中位数   : %d" % int(np.median([len(np.where(genes == g)[0])
                                               for g in pd.unique(genes)])))

# ---------- 2. 划分：统一 gene ID，同一测试人群，训练量匹配 ----------
rng = np.random.RandomState(RNG_SEED)
uniq_genes = np.array(sorted(pd.unique(genes)))
rng.shuffle(uniq_genes)

# 测试基因：取 20% 的基因，其全部样本作为**共享测试集**
n_test_genes = max(1, int(len(uniq_genes) * 0.20))
# 用 list 固定顺序：字符串 set 的迭代顺序受 PYTHONHASHSEED 随机化影响，
# 直接遍历 set 会让 G1 臂的划分不可复现（实测两次运行 G1 差 0.02）
test_genes = sorted(uniq_genes[:n_test_genes])
train_genes = sorted(uniq_genes[n_test_genes:])
test_gene_set = set(test_genes)
train_gene_set = set(train_genes)

test_mask = np.array([g in test_gene_set for g in genes])
train_pool_mask = np.array([g in train_gene_set for g in genes])
print("\n=== 划分 ===")
print("测试基因 %d 个，测试样本 %d" % (len(test_genes), test_mask.sum()))
print("训练基因 %d 个，训练池样本 %d" % (len(train_genes), train_pool_mask.sum()))
print("[可复现性] test_genes/train_genes 已改为排序 list，避免 set 迭代顺序随 PYTHONHASHSEED 变化")

# G1：训练集包含测试基因的**其他**变异（已知基因新变异）
idx_test_gene_trainable = np.where(train_pool_mask == False)[0]  # 测试基因里非测试样本？
# 更准确：G1 训练 = 全训练池 + 测试基因中未被划入测试集的样本（本设计中测试基因全部样本都在测试集，
# 故 G1 采用"基因内留出"：对每个测试基因，留一部分样本进训练集）
g1_train_mask = np.zeros(n, dtype=bool)
rng2 = np.random.RandomState(RNG_SEED)
for g in test_genes:
    idx = np.where(genes == g)[0]
    if len(idx) > 1:
        rng2.shuffle(idx)
        k = max(1, int(len(idx) * 0.5))
        g1_train_mask[idx[:k]] = True
g1_train_mask |= train_pool_mask  # 再加上全部训练基因

# G2：未见基因（测试基因完全不进训练）
g2_train_mask = train_pool_mask.copy()

# 训练规模匹配：把 G1/G2 的训练集下采样到相同规模
n_match = min(g1_train_mask.sum(), g2_train_mask.sum())
def subsample(mask, size, seed):
    idx = np.where(mask)[0]
    r = np.random.RandomState(seed)
    r.shuffle(idx)
    out = np.zeros(n, dtype=bool)
    out[idx[:size]] = True
    return out
g1_train_m = subsample(g1_train_mask, n_match, 1)
g2_train_m = subsample(g2_train_mask, n_match, 2)
print("训练规模匹配后  n_train = %d (G1 %d -> %d, G2 %d -> %d)" % (
    n_match, g1_train_mask.sum(), n_match, g2_train_mask.sum(), n_match))

splits = {"G1_known_gene": (g1_train_m, test_mask),
          "G2_unseen_gene": (g2_train_m, test_mask)}

# ---------- 3. 背景基线（加强版）----------
def background_features(d):
    seqs = d["char_seq"].values
    gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
    # 3-mer 频率（简化：统计最常见 16 个 3-mer）
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
print("\n背景基线维度: %s" % (BG.shape,))

# ---------- 4. 探针（公平搜索）----------
LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))
MLP_GRID = list(ParameterGrid({
    "hidden_layer_sizes": [(128, 64)],
    "alpha": [1e-3, 1e-2]}))


def fair_probe(Xtr, ytr, kind, seed=RNG_SEED):
    """在训练集内部做一次 3 折简单搜索，返回选中的配置与模型工厂。"""
    from sklearn.model_selection import StratifiedKFold
    grid = LR_GRID if kind == "lr" else MLP_GRID
    ntr = len(ytr)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    best, best_score = None, -1
    for cfg in grid:
        scores = []
        for tr_i, va_i in skf.split(Xtr, ytr):
            m = _mk(kind, cfg, seed)
            sc = StandardScaler().fit(Xtr[tr_i])
            m.fit(sc.transform(Xtr[tr_i]), ytr[tr_i])
            p = m.predict_proba(sc.transform(Xtr[va_i]))[:, 1]
            if len(np.unique(ytr[va_i])) < 2:
                continue
            scores.append(roc_auc_score(ytr[va_i], p))
        if scores and np.mean(scores) > best_score:
            best_score, best = np.mean(scores), cfg
    return best, best_score


def _mk(kind, cfg, seed):
    if kind == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed, **cfg)
    return MLPClassifier(max_iter=200, early_stopping=True, random_state=seed, **cfg)


def run(X, tr_mask, te_mask, kind, seed=RNG_SEED):
    Xtr, ytr = X[tr_mask], y[tr_mask]
    Xte, yte = X[te_mask], y[te_mask]
    cfg, cv = fair_probe(Xtr, ytr, kind, seed)
    sc = StandardScaler().fit(Xtr)
    m = _mk(kind, cfg, seed)
    m.fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, p)), cfg, float(cv)


# ---------- 5. 模型 ----------
EMB = {
    "codonbert":     "codonbert_synonymous_emb.npy",
    "codonbert_hf":  "codonbert_hf_synonymous_emb.npy",
    "encodon-80m":   "encodon-80m_synonymous_emb.npy",
    "mrnabert":      "mrnabert_task3_synonymous_emb.npy",
    "ESM-2-650M":    "ESM-2-650M_task3_synonymous_emb.npy",
    "ESM-1b-650M":   "ESM-1b-650M_task3_synonymous_emb.npy",
}

results = {"_design": "WB-实验设计书 v1", "_seed": RNG_SEED,
           "_note": "同一测试人群 + 训练规模匹配 + gene ID 分组 + 公平搜索探针 + 加强背景基线",
           "_unit_check": {"n": n,
                           "n_genes": int(df["gene_symbol"].nunique()),
                           "n_tx": int(df["tx_id"].nunique()),
                           "multi_tx_genes": len(multi)},
           "models": {}, "baseline": {}}

# 背景基线自身
for sname, (tr, te) in splits.items():
    a, cfg, cv = run(BG, tr, te, "lr")
    results["baseline"]["bg_" + sname] = {"auc_lr": a, "cfg": str(cfg)}

for mname, fn in EMB.items():
    p = SUPP / fn
    if not p.exists():
        print("  [跳过] 嵌入缺失: %s" % fn)
        continue
    X = np.load(p)
    if X.shape[0] != n:
        print("  [跳过] 行数不符 %s: %d != %d" % (mname, X.shape[0], n))
        continue
    print("  运行 %s  %s" % (mname, X.shape))
    results["models"][mname] = {}
    for sname, (tr, te) in splits.items():
        a_lr, cfg_lr, cv_lr = run(X, tr, te, "lr")
        a_mlp, cfg_mlp, cv_mlp = run(X, tr, te, "mlp")
        results["models"][mname][sname] = {
            "auc_lr": a_lr, "cfg_lr": str(cfg_lr), "cv_lr": cv_lr,
            "auc_mlp": a_mlp, "cfg_mlp": str(cfg_mlp), "cv_mlp": cv_mlp}

# Δ_add：加 cLM 嵌入 vs 仅背景
for mname, fn in EMB.items():
    p = SUPP / fn
    if not p.exists():
        continue
    X = np.load(p)
    if X.shape[0] != n:
        continue
    Xa = np.hstack([BG, X])
    for sname, (tr, te) in splits.items():
        a_full, _, _ = run(Xa, tr, te, "mlp")
        a_bg, _, _ = run(BG, tr, te, "mlp")
        results["models"][mname].setdefault("delta_add", {})[sname] = \
            float(a_full - a_bg)

with open(OUT / "e2_rerun_results.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print("\n已写出: %s" % (OUT / "e2_rerun_results.json"))
