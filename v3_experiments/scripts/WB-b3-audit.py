# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
B3 可信度审计（2026-10-02）

背景：补跑发现 B3（简单参考—替代密码子特征）在未见基因上有 AUC 0.751，
远高于全部神经模型。这个数字如果成立，会改写整篇的结论，所以必须先排伪特征。

三个必查的疑点：
  Q1 B3 的"构造失败"样本被统一塞进 AAA 列 —— 失败率是否与标签相关？
     若是，B3 就携带一个和生物学无关的伪特征，0.751 不能信。
  Q2 0.751 里有多少来自 cpos（CDS 位置，属**背景**不是变异）？
     必须拆出"纯密码子 one-hot"的成绩。
  Q3 到底是"参考密码子"、"替代密码子"还是"两者的变化"在提供信息？
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"

df = pd.read_parquet(SYN / "synpath_variants.parquet")
y = df["label"].values.astype(int)
genes = df["gene_symbol"].values
n = len(df)
seqs = df["char_seq"].values
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
cpos = df["cpos"].values.reshape(-1, 1).astype(float)

COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def _rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s))


strand_plus = np.array([(len(s) > 45 and s[45].upper() == r)
                        for s, r in zip(seqs, ref_base)])
strand_minus = np.array([(len(s) > 45 and s[45].upper() == COMP.get(r, "N"))
                         for s, r in zip(seqs, ref_base)])
ok_strand = strand_plus | strand_minus

U1 = np.load(OUT / "e1s_usable.npy").astype(bool)
vix_all = np.load(OUT / "e1s_variant_index.npy")
ph_all = np.load(OUT / "e1s_phase_pick.npy")
m = np.where(U1)[0]
vix = (vix_all[m] if len(vix_all) == n else vix_all).astype(int)
ph = (ph_all[m] if len(ph_all) == n else ph_all).astype(int)

BASES = "ACGT"
CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]
CIDX = {c: i for i, c in enumerate(CODONS)}


def build(split_ok=True):
    """返回 (ref_oh, alt_oh, ok_mask)。split_ok=True 时失败样本单独成列。"""
    rc_list, ac_list, ok = [], [], np.zeros(len(m), dtype=bool)
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
            rc_list.append(None); ac_list.append(None); continue
        rb = ref_base[gi] if plus else COMP.get(ref_base[gi], "N")
        ab = alt_base[gi] if plus else COMP.get(alt_base[gi], "N")
        if cod[off] != rb:
            rc_list.append(None); ac_list.append(None); continue
        rc_list.append(cod); ac_list.append(cod[:off] + ab + cod[off + 1:])
        ok[k] = True
    # one-hot：失败样本给一个**独立**的指示列，不再混进 AAA
    R = np.zeros((len(m), 65))
    A = np.zeros((len(m), 65))
    for i, (r, a) in enumerate(zip(rc_list, ac_list)):
        if r is None:
            R[i, 64] = 1.0
            A[i, 64] = 1.0
        else:
            R[i, CIDX[r]] = 1.0
            A[i, CIDX[a]] = 1.0
    return R, A, ok


R, A, ok = build()
ys = y[m]
gs = genes[m]
gc_s = gc[m].reshape(-1, 1)
cpos_s = cpos[m]

print("usable 子集 %d；B3 构造成功 %d (%.1f%%)" % (len(m), ok.sum(), 100 * ok.mean()))

# ---------- Q1：构造失败率是否与标签相关 ----------
p_ok = ys[ok].mean()
p_bad = ys[~ok].mean() if (~ok).sum() else float("nan")
print()
print("Q1 构造成功组的正样本率 %.4f (n=%d)" % (p_ok, ok.sum()))
print("  构造失败组的正样本率 %.4f (n=%d)" % (p_bad, (~ok).sum()))
print("  => 差 %.4f。若接近 0，则『失败』本身不是伪特征；若明显偏离，0.751 要打折"
      % abs(p_ok - p_bad))

# ---------- 评价协议（与主脚本一致：基因留出 5 折 × 5 次） ----------
LR_GRID = list(ParameterGrid({"C": [0.1, 1.0]}))


def _search(Xtr, ytr, seed=42):
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    best, bs = None, -1.0
    for cfg in LR_GRID:
        sc = []
        for a, b in skf.split(Xtr, ytr):
            if len(np.unique(ytr[a])) < 2 or len(np.unique(ytr[b])) < 2:
                continue
            s = StandardScaler().fit(Xtr[a])
            mm = LogisticRegression(max_iter=2000, random_state=seed, **cfg)
            mm.fit(s.transform(Xtr[a]), ytr[a])
            sc.append(roc_auc_score(ytr[b], mm.predict_proba(s.transform(Xtr[b]))[:, 1]))
        if sc and np.mean(sc) > bs:
            bs, best = float(np.mean(sc)), cfg
    return best


def evaluate(X, tag):
    fs = []
    for rep in range(5):
        rng = np.random.RandomState(rep)
        ug = np.array(sorted(pd.unique(gs)))
        rng.shuffle(ug)
        folds = np.array_split(ug, 5)
        g2f = {}
        for fi, fg in enumerate(folds):
            for g in fg:
                g2f[g] = fi
        fof = np.array([g2f[g] for g in gs])
        for fi in range(5):
            te = np.where(fof == fi)[0]
            tr = np.where(fof != fi)[0]
            if len(np.unique(ys[te])) < 2:
                continue
            cfg = _search(X[tr], ys[tr])
            s = StandardScaler().fit(X[tr])
            mm = LogisticRegression(max_iter=2000, random_state=42, **cfg)
            mm.fit(s.transform(X[tr]), ys[tr])
            fs.append(roc_auc_score(ys[te], mm.predict_proba(s.transform(X[te]))[:, 1]))
    a = np.asarray(fs)
    print("  %-28s 逐折 %.4f ± %.4f" % (tag, a.mean(), a.std(ddof=1)))
    return float(a.mean())


print()
print("Q2/Q3：拆解 0.751 的来源（同一协议，usable 子集，基因留出）")
print("-" * 60)
cands = {
    "B3 全量(128+GC+cpos)": np.hstack([R, A, gc_s, cpos_s]),
    "仅 ref 密码子 one-hot": R,
    "仅 alt 密码子 one-hot": A,
    "ref+alt（去掉 GC/cpos）": np.hstack([R, A]),
    "仅 GC + cpos（纯背景）": np.hstack([gc_s, cpos_s]),
}
out = {}
for k, X in cands.items():
    out[k] = evaluate(X, k)
print("-" * 60)
print("⇒ 纯密码子对（无 GC/cpos）: %.4f" % out["ref+alt（去掉 GC/cpos）"])
print("⇒ 纯背景（GC+cpos）      : %.4f" % out["仅 GC + cpos（纯背景）"])
print("⇒ 密码子对相对背景的增量  : %+.4f"
      % (out["ref+alt（去掉 GC/cpos）"] - out["仅 GC + cpos（纯背景）"]))
