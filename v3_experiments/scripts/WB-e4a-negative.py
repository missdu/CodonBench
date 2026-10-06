# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e4a-negative.py (2026-10-03)
E4-A 的负例：证明"置换后 B3 仍有 0.6914"不是评估流程泄漏造成的。

三种标签条件，同一特征（B3）、同一折（240 folds）、同一探针：
  A real         真实标签                                   ⇒ 已知 0.7740
  B gene_perm    训练集内**按基因**置换（保留基因阳性率）    ⇒ 已知 0.6914
  C global_perm  训练集内**全局**随机置换（连基因阳性率也破）⇒ 负例，期望 ≈ 0.500

判据：
  若 C ≈ 0.500 ⇒ 评估流程无泄漏，B 的 0.6914 是真实信号（基因级标签结构）
  若 C 明显 > 0.500 ⇒ 有泄漏（例如测试集信息进了训练、或特征本身编码了标签）

  同时期望 A > B > C 严格单调：
    B − C = 基因标签结构能支撑的部分
    A − B = 变异特异部分

另加一个更强的负例：
  D test_perm    训练集真实、测试集标签置换 ⇒ 期望 ≈ 0.500
    （若 ≠ 0.5，说明折内有跨样本信息串漏）
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
SEED = 42

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
y = df["label"].values.astype(int); genes = df["gene_symbol"].values; n = len(df)
CENTER = 45
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = df["char_seq"].values
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
cpos = df["cpos"].values.reshape(-1, 1).astype(float)
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
CIDX = {a + b + c: i for i, (a, b, c) in enumerate(
    (x, y_, z) for x in "ACGT" for y_ in "ACGT" for z in "ACGT")}
CIDX["NNN"] = len(CIDX)
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
# 🔴 相位只由 CDS 位置决定：off = (cpos - 1) % 3（不用 e1s_phase_pick.npy）
cpos_i = df["cpos"].values.astype(int)
off_arr = (cpos_i - 1) % 3
# 🔴 与基线脚本同步（2026-10-03 第三次修）：char_seq 已是编码链，**不反向互补**；
#    只对负链基因的碱基取互补。链向以 HGVS c. 碱基为基准。
import re as _re
_PATC = _re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
_hg_ref = []
for _nm in df["Name"]:
    _m = _PATC.search(str(_nm))
    _hg_ref.append(_m.group(2) if _m else None)
plus = np.array([(_hg_ref[i] is not None and _hg_ref[i] == ref_base[i]) for i in range(n)])
print(f"链向(以 HGVS 为基准): plus={int(plus.sum())} minus={int(n - plus.sum())}")
rc, ac = [], []
for i in range(n):
    s = str(seqs[i]).upper()
    vi = int(vix[i]); off = int(off_arr[i]); st = vi - off
    cod = s[st:st + 3]
    rb = ref_base[i] if plus[i] else COMP.get(ref_base[i], "N")
    ab = alt_base[i] if plus[i] else COMP.get(alt_base[i], "N")
    if len(cod) != 3 or not (0 <= off < 3) or cod[off] != rb:
        rc.append("NNN"); ac.append("NNN"); continue
    rc.append(cod); ac.append(cod[:off] + ab + cod[off + 1:])
def oh(a):
    M = np.zeros((len(a), 65))
    for i, c in enumerate(a): M[i, CIDX.get(c, CIDX["AAA"])] = 1.0
    return M
B3 = np.hstack([oh(rc), oh(ac), gc.reshape(-1, 1), cpos])

d = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fid = d["fold_id"]; gidx = d["global_index"]; N240 = int(fid.max()) + 1

def fit(Xtr, ytr, seed=SEED):
    if len(set(ytr)) < 2: return None
    try:
        sp = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(Xtr, ytr))
    except Exception:
        sp = []
    if not sp: sp = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    best, bs = None, -1
    for C in (0.1, 1.0):
        ss = []
        for a, b in sp:
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))
            p.fit(Xtr[a], ytr[a]); pv = p.predict_proba(Xtr[b])[:, 1]
            ss.append(0.5 if len(set(ytr[b])) < 2 else roc_auc_score(ytr[b], pv))
        s = float(np.mean(ss))
        if s > bs: bs, best = s, C
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=best))
    p.fit(Xtr, ytr); return p

def run(mode):
    rng = np.random.RandomState(SEED)
    aucs = []
    for k in range(N240):
        te = gidx[fid == k]; tr = np.setdiff1d(np.arange(n), te)
        ytr = y[tr].copy(); yte = y[te].copy()
        if mode == "gene_perm":
            for g in np.unique(genes[tr]):
                m = genes[tr] == g
                if m.sum() > 1:
                    idx = np.where(m)[0]; ytr[idx] = rng.permutation(ytr[idx])
        elif mode == "global_perm":
            ytr = rng.permutation(ytr)
        elif mode == "test_perm":
            yte = rng.permutation(yte)
        p = fit(B3[tr], ytr)
        if p is None: aucs.append(np.nan); continue
        s = p.predict_proba(B3[te])[:, 1]
        aucs.append(roc_auc_score(yte, s) if len(set(yte)) > 1 else np.nan)
    a = np.array(aucs, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) / np.sqrt(np.isfinite(a).sum()))

print("B3 / 240 valid folds / LR  —— 四种标签条件")
print(f"{'条件':16s} {'逐折AUC':>9s} {'SEM':>8s}")
res = {}
for m in ("real", "gene_perm", "global_perm", "test_perm"):
    mu, se = run(m); res[m] = {"auc": mu, "sem": se}
    print(f"  {m:14s} {mu:9.4f} {se:8.4f}")

print()
ok_leak = abs(res["global_perm"]["auc"] - 0.5) < 0.03
ok_test = abs(res["test_perm"]["auc"] - 0.5) < 0.03
mono = res["real"]["auc"] > res["gene_perm"]["auc"] > res["global_perm"]["auc"]
print(f"负例1 global_perm ≈ 0.5 ? {'✅ PASS' if ok_leak else '❌ FAIL 评估流程有泄漏'}  ({res['global_perm']['auc']:.4f})")
print(f"负例2 test_perm   ≈ 0.5 ? {'✅ PASS' if ok_test else '❌ FAIL 折内有串漏'}  ({res['test_perm']['auc']:.4f})")
print(f"单调 real > gene_perm > global_perm ? {'✅' if mono else '❌'}")
print()
print("分解（相对 0.5 的超额性能）:")
ex = res["real"]["auc"] - 0.5
print(f"  基因标签结构支撑 = (gene_perm − global_perm)/超额 = "
      f"{(res['gene_perm']['auc']-res['global_perm']['auc'])/ex:.3f}")
print(f"  变异特异部分     = (real − gene_perm)/超额 = "
      f"{(res['real']['auc']-res['gene_perm']['auc'])/ex:.3f}")

import json
json.dump({"B3_240folds": res, "checks": {"global_perm_ok": bool(ok_leak),
           "test_perm_ok": bool(ok_test), "monotone": bool(mono)}},
          open(f"{OUT}/e4a_negative_v1.json", "w"), indent=2, ensure_ascii=False)
print("\n已写入", f"{OUT}/e4a_negative_v1.json")
