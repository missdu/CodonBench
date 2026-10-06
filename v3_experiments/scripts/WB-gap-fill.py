# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB 缺口补齐（2026-10-02）—— 按「新论文设计GPT.md」第五章与第七章补齐我这版没做的部分。

补的六件事（都不需要 GPU）：
  1. 基线阶梯 B0/B1/B2/B1+B2   —— 尤其 B1「训练基因标签率」，
                                  这是**直接**测量基因身份的基线；
                                  我之前只从神经模型间接推断，没有直接测。
  2. B3 简单参考—替代密码子特征  —— 不依赖预训练的变异对照（GPT: E1 的 B3）。
  3. B3 + C-diff                —— cLM 相对**强简单基线**的增量（GPT: B3+C）。
  4. E5 探针公平评价            —— LR 与 MLP 各自内部搜索、同一预算（GPT: E5-A）。
  5. 逐折汇总为主 + pooled 为辅  —— GPT §七.3 要求，我之前只有 pooled。
  6. 基因内排序 AUC 及覆盖       —— GPT §七.2 关键辅助指标；同时也是
                                  GPT §四.2「组内—组间分解」的输入。
  7. MDES 按估计目标分别算       —— GPT §七.5 明确要求**删除**
                                  「一个 9.3 pp 覆盖所有比较」的思路。

另外：E4-B（同位点不同同义替代的配对评价）先只报**可用配对数**，
数量不足就按 GPT 的要求作描述性分析，不强凑结果。
"""
import json, os, sys
from collections import Counter
from pathlib import Path

# ONLY_SUB=1：跳过全量主循环与 E5，只跑 usable 子集段与 E4-B（用于快速取子集数字）
ONLY_SUB = os.environ.get("ONLY_SUB", "0") == "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"
OUT.mkdir(parents=True, exist_ok=True)

N_REPEAT = 5
N_FOLD = 5
N_BOOT = 1000
Z_MDES = 2.8016  # z_{alpha/2}+z_beta, alpha=0.05 双侧, power=0.80

# ---------------- 1. 数据 ----------------
df = pd.read_parquet(SYN / "synpath_variants.parquet")
y = df["label"].values.astype(int)
genes = df["gene_symbol"].values
n = len(df)
print("样本 %d | 基因 %d | 正样本 %d (%.3f)"
      % (n, len(pd.unique(genes)), y.sum(), y.mean()))

# ---------------- 2. 基线阶梯 ----------------
seqs = df["char_seq"].values
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])

cnt = Counter()
for s in seqs:
    for i in range(len(s) - 2):
        cnt[s[i:i + 3]] += 1
TOP_KMER = [k for k, _ in cnt.most_common(16)]
kmer = np.array([[s.count(k) / max(1, len(s)) for k in TOP_KMER] for s in seqs])
cpos = df["cpos"].values.reshape(-1, 1).astype(float)

B2 = np.hstack([gc.reshape(-1, 1), kmer, cpos])          # 背景结构，18 维
print("B2 背景特征维度:", B2.shape)

# B3：简单参考—替代密码子特征（不依赖预训练）
# 注意：ReferenceAllele / AlternateAllele 两列在 SynPath 里**全为 'na'**，
# 真值在 VCF 列里。用错列会让链向判定全军覆没（实测 0/2840 可判定）。
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
# 链向：窗口中心碱基与参考碱基一致 => 正链；互补 => 负链（与 E1 严格版同一规则）。
# 注意 char_seq 是小写，比较前必须统一大小写。
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def _rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s))


CENTER = 45
strand_plus = np.array([
    (len(s) > CENTER and s[CENTER].upper() == r) for s, r in zip(seqs, ref_base)])
strand_minus = np.array([
    (len(s) > CENTER and s[CENTER].upper() == COMP.get(r, "N"))
    for s, r in zip(seqs, ref_base)])
print("链向：正链 %d / 负链 %d / 不可判定 %d"
      % (strand_plus.sum(), strand_minus.sum(), n - strand_plus.sum() - strand_minus.sum()))

BASES = "ACGT"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
CIDX = {c: i for i, c in enumerate(CODONS)}


def _onehot(arr):
    M = np.zeros((len(arr), 64))
    for i, c in enumerate(arr):
        M[i, CIDX.get(c, CIDX["AAA"])] = 1.0
    return M


def build_B3(mask):
    """在给定行掩码上构造 B3：不依赖预训练的简单参考—替代密码子特征。

    链向用本脚本的判定（char_seq 中心 vs VCF 参考碱基），
    阅读框与变异位置用 E1 严格版**已验证**的判定（e1s_phase_pick / e1s_variant_index）。
    只有当变异确实落在密码子内、且替换后仍是合法三联体时才构造成功。
    """
    vix_a = np.load(OUT / "e1s_variant_index.npy")
    ph_a = np.load(OUT / "e1s_phase_pick.npy")
    m = np.where(mask)[0]
    # 这两个文件是**全量**数组（长度 n），要按 usable 掩码索引；
    # 而 e1s_*_ref.npy 等嵌入文件已是子集（长度 = usable 数），不能再过滤。
    vix = vix_a[m] if len(vix_a) == n else vix_a
    ph = ph_a[m] if len(ph_a) == n else ph_a
    assert len(vix) == len(m) and len(ph) == len(m), \
        "E1 的 variant_index/phase 行数与 usable 数不一致"
    rc_list, ac_list, ok = [], [], 0
    for k, gi in enumerate(m):
        s = str(seqs[gi]).upper()
        plus = bool(strand_plus[gi])
        if not plus:
            s = _rc(s)
        vi = int(vix[k])
        start = vi - ((vi - int(ph[k])) % 3)
        cod = s[start:start + 3]
        off = vi - start
        if len(cod) != 3 or not (0 <= off < 3):
            rc_list.append("NNN")
            ac_list.append("NNN")
            continue
        rb = ref_base[gi] if plus else COMP.get(ref_base[gi], "N")
        ab = alt_base[gi] if plus else COMP.get(alt_base[gi], "N")
        if cod[off] != rb:      # 链/框判定不一致，宁可放弃
            rc_list.append("NNN")
            ac_list.append("NNN")
            continue
        rc_list.append(cod)
        ac_list.append(cod[:off] + ab + cod[off + 1:])
        ok += 1
    X = np.hstack([_onehot(rc_list), _onehot(ac_list),
                   gc[m].reshape(-1, 1), cpos[m]])
    print("B3 构造成功 %d / %d；维度 %s" % (ok, len(m), X.shape))
    return X


def gene_rate_features(tr_mask):
    """B1：训练集内每个基因的标签率（带平滑）；未见基因回退到全局率。"""
    g_tr, y_tr = genes[tr_mask], y[tr_mask]
    glob = y_tr.mean()
    num = pd.Series(y_tr).groupby(g_tr).sum()
    den = pd.Series(y_tr).groupby(g_tr).count()
    # 平滑：向全局率收缩，k=5
    rate = (num + 5 * glob) / (den + 5)
    m = np.array([rate.get(g, glob) for g in genes])
    return m.reshape(-1, 1)


# ---------------- 3. 嵌入 ----------------
EMB = {
    "codonbert":    "codonbert_synonymous_emb.npy",
    "codonbert_hf": "codonbert_hf_synonymous_emb.npy",
    "encodon-80m":  "encodon-80m_synonymous_emb.npy",
    "mrnabert":     "mrnabert_task3_synonymous_emb.npy",
    "ESM-2-650M":   "ESM-2-650M_task3_synonymous_emb.npy",
    "ESM-1b-650M":  "ESM-1b-650M_task3_synonymous_emb.npy",
}
Xs = {}
for mname, fn in EMB.items():
    p = SUPP / fn
    if p.exists():
        X = np.load(p)
        if X.shape[0] == n:
            Xs[mname] = X
print("可用嵌入:", list(Xs))

# E1 严格版的参考/替代嵌入与差分向量。
# 它们只在 E1 判定的 usable 子集上有定义（链向可定 + 阅读框可定），
# 因此**不并入全量主循环**，单独在子集内做对比，避免口径污染。
p_ref = OUT / "e1s_codonbert_ref.npy"
E1 = {}
if p_ref.exists():
    U1 = np.load(OUT / "e1s_usable.npy").astype(bool)
    R = np.asarray(np.load(OUT / "e1s_codonbert_ref.npy", allow_pickle=True), dtype=float)
    A = np.asarray(np.load(OUT / "e1s_codonbert_alt.npy", allow_pickle=True), dtype=float)
    # 注意：这两个文件**已经是 usable 子集**（长度 = U1.sum()），不能再按 U1 过滤一次。
    # 顺序与 [i for i in range(n) if U1[i]] 一致（按原 df 顺序筛出）。
    assert R.shape[0] == int(U1.sum()), \
        "E1 嵌入行数 %d 与 usable 数 %d 不符" % (R.shape[0], int(U1.sum()))
    E1 = {"mask": U1, "ref": R, "alt": A, "diff": A - R}
    print("载入 E1 严格版嵌入: usable %d | diff %s" % (int(U1.sum()), E1["diff"].shape))

# ---------------- 4. 探针（公平预算） ----------------
LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))
MLP_GRID = list(ParameterGrid({"hidden_layer_sizes": [(128, 64)],
                               "alpha": [1e-3, 1e-2]}))


def _mk(kind, cfg, seed):
    if kind == "lr":
        return LogisticRegression(max_iter=2000, random_state=seed, **cfg)
    return MLPClassifier(max_iter=200, early_stopping=True, random_state=seed, **cfg)


def _search(Xtr, ytr, kind, seed=42):
    """训练集内部 2 折搜索；两个探针家族拥有**可比较**的配置数（各 2 个）。"""
    grid = LR_GRID if kind == "lr" else MLP_GRID
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    best, best_sc = None, -1.0
    for cfg in grid:
        sc = []
        for a, b in skf.split(Xtr, ytr):
            if len(np.unique(ytr[a])) < 2 or len(np.unique(ytr[b])) < 2:
                continue
            s = StandardScaler().fit(Xtr[a])
            m = _mk(kind, cfg, seed)
            m.fit(s.transform(Xtr[a]), ytr[a])
            sc.append(roc_auc_score(ytr[b], m.predict_proba(s.transform(Xtr[b]))[:, 1]))
        if sc and np.mean(sc) > best_sc:
            best_sc, best = float(np.mean(sc)), cfg
    return best


def _fit_score(X, tr, te, kind, seed=42):
    Xtr, ytr = X[tr], y[tr]
    cfg = _search(Xtr, ytr, kind, seed)
    s = StandardScaler().fit(Xtr)
    m = _mk(kind, cfg, seed)
    m.fit(s.transform(Xtr), ytr)
    return m.predict_proba(s.transform(X[te]))[:, 1]


def _fit_local(X, tr, te, yv, kind="lr", seed=42):
    """与 _fit_score 相同，但用于子集（标签向量不是全局 y）。"""
    Xtr, ytr = X[tr], yv[tr]
    cfg = _search(Xtr, ytr, kind, seed)
    s = StandardScaler().fit(Xtr)
    m = _mk(kind, cfg, seed)
    m.fit(s.transform(Xtr), ytr)
    return m.predict_proba(s.transform(X[te]))[:, 1]


def within_gene_auc(score, te_idx):
    """基因内排序 AUC（macro，等权每个基因）+ 覆盖率。
    只保留测试集中正负样本都 >=1 的基因。"""
    gs = genes[te_idx]
    ss = score[te_idx]
    yy = y[te_idx]
    aucs, w = [], []
    for g in pd.unique(gs):
        m = gs == g
        if yy[m].sum() >= 1 and (1 - yy[m]).sum() >= 1:
            aucs.append(roc_auc_score(yy[m], ss[m]))
            w.append(int(m.sum()))
    if not aucs:
        return None, 0, 0
    return float(np.mean(aucs)), len(aucs), int(sum(w))


# ---------------- 5. 主循环 ----------------
all_genes = np.array(sorted(pd.unique(genes)))
# 特征集：名 -> (构造函数, 是否依赖训练集)
# B3 不在全量主循环里：它依赖链向与阅读框，只在 E1 验证过的 usable 子集上有定义。
FEATS = ["B0_const", "B1_gene_rate", "B2_background", "B1+B2"]


def build(kind_name, tr_mask):
    if kind_name == "B0_const":
        return np.full((n, 1), y[tr_mask].mean(), dtype=float)
    if kind_name == "B1_gene_rate":
        return gene_rate_features(tr_mask)
    if kind_name == "B2_background":
        return B2
    if kind_name == "B1+B2":
        return np.hstack([gene_rate_features(tr_mask), B2])
    raise KeyError(kind_name)


res = {}
for key in FEATS + list(Xs.keys()):
    # by_rep：按"独立基因划分"分组存折内 AUC。
    # 独立单位是**划分**不是折：同一次划分的 5 折共享训练集，不能当 5 个独立样本。
    res[key] = {"by_rep": [], "fold": [], "pooled": None,
                "within": [], "cov_gene": [], "cov_n": []}

pooled_scores = {k: np.zeros(n) for k in res}
pooled_seen = np.zeros(n, dtype=bool)

for rep in range(0 if ONLY_SUB else N_REPEAT):
    rng = np.random.RandomState(rep)
    sh = all_genes.copy()
    rng.shuffle(sh)
    folds = np.array_split(sh, N_FOLD)
    # 按基因分折
    gene2fold = {}
    for fi, fg in enumerate(folds):
        for g in fg:
            gene2fold[g] = fi
    fold_of = np.array([gene2fold[g] for g in genes])
    fold_scores = {k: [] for k in res}

    for fi in range(N_FOLD):
        te = np.where(fold_of == fi)[0]
        tr = np.where(fold_of != fi)[0]
        if len(np.unique(y[te])) < 2:
            continue
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[tr] = True

        for key in res:
            if key == "B3+diff" and "codonbert_diff" not in Xs:
                continue
            X = build(key, tr_mask) if key in FEATS else Xs[key]
            p = _fit_score(X, tr, te, "lr", seed=42)
            a = float(roc_auc_score(y[te], p))
            res[key]["fold"].append(a)
            fold_scores[key].append(a)
            full = np.zeros(n)
            full[te] = p
            w, ng, nn = within_gene_auc(full, te)
            if w is not None:
                res[key]["within"].append(w)
                res[key]["cov_gene"].append(ng)
                res[key]["cov_n"].append(nn)
            pooled_scores[key][te] = p
        pooled_seen[te] = True
    for k in res:
        res[k]["by_rep"].append(fold_scores[k])
    print("  rep %d done" % rep, flush=True)

# pooled（辅助指标）
for key in res:
    if pooled_seen.all():
        res[key]["pooled"] = float(roc_auc_score(y, pooled_scores[key]))

# ---------------- 6. 汇总 ----------------
summary = {}
for key, v in res.items():
    f = np.asarray(v["fold"], dtype=float)
    if len(f) == 0:
        continue
    summary[key] = {
        "fold_mean": round(float(f.mean()), 4),
        "fold_sd": round(float(f.std(ddof=1)), 4) if len(f) > 1 else None,
        "n_folds": int(len(f)),
        "fold_sem": round(float(f.std(ddof=1) / np.sqrt(len(f))), 4) if len(f) > 1 else None,
        "pooled_oof": round(float(v["pooled"]), 4) if v["pooled"] is not None else None,
        "within_gene_macro": round(float(np.mean(v["within"])), 4) if v["within"] else None,
        "within_gene_sd": round(float(np.std(v["within"], ddof=1)), 4) if len(v["within"]) > 1 else None,
        "cov_genes_per_fold": round(float(np.mean(v["cov_gene"])), 1) if v["cov_gene"] else None,
    }
    # MDES：独立单位是**基因划分**（5 次），不是折（25 个）。
    # 同一次划分的 5 折共享训练集，把折当独立样本会把 SEM 低估约 sqrt(5) 倍
    # （GPT §七.1：「不能用增加种子代替增加独立基因」）。
    reps = [float(np.mean(r)) for r in v["by_rep"] if len(r)]
    if len(reps) > 1:
        sem_indep = float(np.std(reps, ddof=1) / np.sqrt(len(reps)))
        summary[key]["rep_means"] = [round(x, 4) for x in reps]
        summary[key]["SE_pp"] = round(sem_indep * 100, 3)
        summary[key]["MDES_pp"] = round(Z_MDES * sem_indep * 100, 2)
        # 折级 SEM 会低估，只作对照，不得用于功效声明
        summary[key]["SE_pp_foldlevel_UNDERSTATED"] = round(
            float(f.std(ddof=1) / np.sqrt(len(f)) * 100), 3)

print()
print("=" * 78)
print("缺口补齐结果（基因留出 5 折 × %d 次重复；主指标 = 逐折均值 ± SD）" % N_REPEAT)
print("=" * 78)
print("%-18s %8s %8s %8s %8s %8s %8s"
      % ("特征/模型", "逐折均值", "SD", "pooled", "基因内", "SE(pp)", "MDES"))
for key, v in summary.items():
    print("%-18s %8.4f %8s %8s %8s %8s %8s"
          % (key, v["fold_mean"],
             v["fold_sd"] if v["fold_sd"] is not None else "-",
             v["pooled_oof"] if v["pooled_oof"] is not None else "-",
             v["within_gene_macro"] if v["within_gene_macro"] is not None else "-",
             v["SE_pp"] if "SE_pp" in v else "-",
             v["MDES_pp"] if "MDES_pp" in v else "-"))
print("（SE/MDES 的独立单位 = 5 次基因划分，非 25 个折）")

# ---------------- 7. E5：探针公平评价（LR vs MLP） ----------------
print()
print("=" * 78)
print("E5 探针公平评价：LR 与 MLP 各自内部搜索、同一配置预算（各 2 个配置）")
print("=" * 78)
probe_cmp = {}
for key in ([] if ONLY_SUB else ["B2_background", "B1+B2"] + list(Xs.keys())):
    probe_cmp[key] = {}
    for kind in ["lr", "mlp"]:
        fs = []
        rng = np.random.RandomState(0)
        sh = all_genes.copy()
        rng.shuffle(sh)
        folds = np.array_split(sh, N_FOLD)
        gene2fold = {}
        for fi, fg in enumerate(folds):
            for g in fg:
                gene2fold[g] = fi
        fold_of = np.array([gene2fold[g] for g in genes])
        for fi in range(N_FOLD):
            te = np.where(fold_of == fi)[0]
            tr = np.where(fold_of != fi)[0]
            if len(np.unique(y[te])) < 2:
                continue
            tr_mask = np.zeros(n, dtype=bool)
            tr_mask[tr] = True
            X = build(key, tr_mask) if key in FEATS else Xs[key]
            p = _fit_score(X, tr, te, kind, seed=42)
            fs.append(float(roc_auc_score(y[te], p)))
        probe_cmp[key][kind] = {
            "fold_mean": round(float(np.mean(fs)), 4),
            "fold_sd": round(float(np.std(fs, ddof=1)), 4) if len(fs) > 1 else None,
        }
    a = probe_cmp[key]["lr"]["fold_mean"]
    b = probe_cmp[key]["mlp"]["fold_mean"]
    probe_cmp[key]["mlp_minus_lr"] = round(b - a, 4)
    print("%-18s LR %.4f | MLP %.4f | MLP-LR %+.4f"
          % (key, a, b, b - a))

# ---------------- 7.5 usable 子集内：cLM 相对**强简单基线**的增量 ----------------
# GPT §五 E1 的 B3+C：测 cLM 变异表征相对"不依赖预训练的变异特征"的增量。
# 只在 E1 判定的 usable 子集内跑，四个输入口径一致。
sub = {}
if E1:
    print()
    print("=" * 78)
    print("usable 子集内：B3（简单变异特征） vs C-diff（cLM 响应向量） vs B3+C-diff")
    print("=" * 78)
    m1 = E1["mask"]
    idx = np.where(m1)[0]
    ys = y[idx]
    gs = genes[idx]
    B3s = build_B3(m1)
    Ds = E1["diff"]
    Rs = E1["ref"]
    SUB = {"B3_codon": B3s, "C_ref": Rs, "C_diff": Ds,
           "B3+C_diff": np.hstack([B3s, Ds])}
    ugenes = np.array(sorted(pd.unique(gs)))
    sub = {k: [] for k in SUB}
    for rep in range(N_REPEAT):
        rng = np.random.RandomState(rep)
        sh = ugenes.copy()
        rng.shuffle(sh)
        folds = np.array_split(sh, N_FOLD)
        g2f = {}
        for fi, fg in enumerate(folds):
            for g in fg:
                g2f[g] = fi
        fof = np.array([g2f[g] for g in gs])
        for fi in range(N_FOLD):
            te = np.where(fof == fi)[0]
            tr = np.where(fof != fi)[0]
            if len(np.unique(ys[te])) < 2:
                continue
            for k, X in SUB.items():
                p = _fit_local(X, tr, te, ys, seed=42)
                sub[k].append(float(roc_auc_score(ys[te], p)))
    print("%-12s %10s %10s" % ("输入", "逐折均值", "SD"))
    for k, v in sub.items():
        a = np.asarray(v, dtype=float)
        print("%-12s %10.4f %10.4f" % (k, a.mean(), a.std(ddof=1)))
    a3 = np.asarray(sub["B3_codon"], float)
    ad = np.asarray(sub["C_diff"], float)
    a33 = np.asarray(sub["B3+C_diff"], float)
    print()
    print("C_diff - B3           : %+.4f" % (ad.mean() - a3.mean()))
    print("(B3+C_diff) - B3      : %+.4f   <- cLM 相对强简单基线的增量"
          % (a33.mean() - a3.mean()))


# ---------------- 8. E4-B：同位点不同同义替代的可用配对数 ----------------
print()
print("=" * 78)
print("E4-B 同位点不同同义替代配对（先只报数量，不足则作描述性分析）")
print("=" * 78)
key4 = pd.Series(list(zip(df["gene_symbol"], df["cpos"], df["codon_seq"])))
grp = key4.groupby(key4).size()
n_multi = int((grp >= 2).sum())
n_pairs = int(sum(v * (v - 1) // 2 for v in grp if v >= 2))
# 标签不同的配对才是可判的
lab = df["label"].values
diff_pairs = 0
for _, idx in key4.groupby(key4).groups.items():
    ii = list(idx)
    if len(ii) < 2:
        continue
    for a in range(len(ii)):
        for b in range(a + 1, len(ii)):
            if lab[ii[a]] != lab[ii[b]]:
                diff_pairs += 1
print("同位点同密码子的变异组: %d 组（>=2 条的 %d 组）" % (len(grp), n_multi))
print("同位点配对总数: %d；其中**标签不同**（可判）的配对: %d" % (n_pairs, diff_pairs))
e4b = {"groups": int(len(grp)), "groups_ge2": n_multi,
       "pairs_total": n_pairs, "pairs_label_discordant": diff_pairs,
       "verdict": ("充足，可做配对评价" if diff_pairs >= 100
                   else "不足，按 GPT 要求作描述性分析，不强凑")}

# ---------------- 9. 输出 ----------------
out = {
    "_design": "WB 缺口补齐：基线阶梯 B0/B1/B2/B3、E5 探针公平评价、逐折汇总、基因内AUC、MDES、E4-B 计数",
    "_source": "新论文设计GPT.md §五(E1/E5/E4-B) 与 §七(统计设计)",
    "_n_repeat": N_REPEAT, "_n_fold": N_FOLD, "_probe_budget": "LR 2 配置 / MLP 2 配置，各自训练集内部 2 折搜索",
    "_main_metric": "逐折 AUC 均值 ± SD（pooled OOF 仅作辅助）",
    "summary": summary,
    "probe_comparison": probe_cmp,
    "subset_B3_vs_cdiff": ({k: {"fold_mean": round(float(np.mean(v)), 4),
                                "fold_sd": round(float(np.std(v, ddof=1)), 4)}
                            for k, v in sub.items()} if sub else None),
    "E4B_paired_sites": e4b,
    "strand": {"plus": int(strand_plus.sum()), "minus": int(strand_minus.sum()),
               "undetermined": int(n - strand_plus.sum() - strand_minus.sum())},
    "note_ref_allele_column": "ReferenceAllele/AlternateAllele 全为 'na'，真值来自 *VCF 列",
}
json.dump(out, open(OUT / "gapfill_results.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print()
print("已写出:", OUT / "gapfill_results.json")
