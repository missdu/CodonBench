# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-b3-ablation.py (2026-10-03)
B3 = 0.8358 之后必须回答的问题：**这个信号从哪来？**
0.8358 在留一基因下是很高的数，不拆开看就等于接受一个黑箱。
本脚本把 B3 拆成互斥的组分，看各自贡献多少。

五个配置（共用同一折、同一探针、同一预算）：
  full      ref one-hot + alt one-hot + GC + cpos    = 0.8358（基准）
  refonly   ref one-hot + GC + cpos                  —— 只给"原来是什么密码子"
  altonly   alt one-hot + GC + cpos                  —— 只给"变成了什么密码子"
  delta     (alt − ref) one-hot + GC + cpos          —— 只给"变化量"，抹掉绝对身份
  nocpos    ref + alt one-hot + GC                   —— 去掉 CDS 位置

读法：
  refonly ≈ altonly ≈ 高 ⇒ 靠的是密码子**身份/使用频率**，不是"变化"
  delta 明显高于两者 ⇒ 真的是"这个变化"本身携带信号
  nocpos 掉很多 ⇒ 位置在提供额外信息（须警惕：位置可能与基因长度/结构相关）
"""
import os, json, re
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
y = df["label"].values.astype(int); n = len(df)
ref_b = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_b = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = np.array([str(s) for s in df["char_seq"].values])
cpos = df["cpos"].values.reshape(-1, 1).astype(float)
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs]).reshape(-1, 1)
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
cpos_i = df["cpos"].values.astype(int)
off_arr = (cpos_i - 1) % 3

COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
CIDX = {a + b + c: i for i, (a, b, c) in enumerate(
    (x, y_, z) for x in "ACGT" for y_ in "ACGT" for z in "ACGT")}
CIDX["NNN"] = len(CIDX)
_PATC = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
hg_ref = []
for nm in df["Name"]:
    m = _PATC.search(str(nm))
    hg_ref.append(m.group(2) if m else None)
plus = np.array([(hg_ref[i] is not None and hg_ref[i] == ref_b[i]) for i in range(n)])

rc_l, ac_l = [], []
for i in range(n):
    s = seqs[i].upper()
    vi = int(vix[i]); off = int(off_arr[i]); st = vi - off
    cod = s[st:st + 3]
    rb = ref_b[i] if plus[i] else COMP.get(ref_b[i], "N")
    ab = alt_b[i] if plus[i] else COMP.get(alt_b[i], "N")
    if len(cod) != 3 or not (0 <= off < 3) or cod[off] != rb:
        rc_l.append("NNN"); ac_l.append("NNN"); continue
    rc_l.append(cod); ac_l.append(cod[:off] + ab + cod[off + 1:])

def oh(a):
    M = np.zeros((len(a), 65))
    for i, c in enumerate(a):
        M[i, CIDX.get(c, CIDX["AAA"])] = 1.0
    return M
R, A = oh(rc_l), oh(ac_l)
print(f"ref/alt one-hot {R.shape}")

FEATS = {
    "full":    np.hstack([R, A, gc, cpos]),
    "refonly": np.hstack([R, gc, cpos]),
    "altonly": np.hstack([A, gc, cpos]),
    "delta":   np.hstack([A - R, gc, cpos]),
    "nocpos":  np.hstack([R, A, gc]),
}

d = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fid = d["fold_id"]; gidx = d["global_index"]; N240 = int(fid.max()) + 1

def fit(Xtr, ytr, seed=SEED):
    if len(set(ytr)) < 2:
        return None
    try:
        sp = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(Xtr, ytr))
    except Exception:
        sp = []
    if not sp:
        sp = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    best, bs = None, -1
    for C in (0.1, 1.0):
        ss = []
        for a, b in sp:
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))
            p.fit(Xtr[a], ytr[a]); pv = p.predict_proba(Xtr[b])[:, 1]
            ss.append(0.5 if len(set(ytr[b])) < 2 else roc_auc_score(ytr[b], pv))
        s = float(np.mean(ss))
        if s > bs:
            bs, best = s, C
    p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=best))
    p.fit(Xtr, ytr)
    return p

res = {}
print(f"\n{'配置':10s} {'维度':>5s} {'逐折AUC':>9s} {'SEM':>8s}")
for name, X in FEATS.items():
    aucs = []
    for k in range(N240):
        te = gidx[fid == k]; tr = np.setdiff1d(np.arange(n), te)
        p = fit(X[tr], y[tr])
        if p is None:
            aucs.append(np.nan); continue
        s = p.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s) if len(set(y[te])) > 1 else np.nan)
    a = np.array(aucs, dtype=float)
    m = np.isfinite(a)
    mu = float(a[m].mean()); se = float(a[m].std(ddof=1) / np.sqrt(m.sum()))
    res[name] = {"dim": int(X.shape[1]), "auc": mu, "sem": se,
                 "auc_per_fold": a.tolist()}
    print(f"  {name:8s} {X.shape[1]:5d} {mu:9.4f} {se:8.4f}")

base = res["full"]["auc"]
print(f"\n相对 full({base:.4f}) 的落差：")
for k in ("refonly", "altonly", "delta", "nocpos"):
    print(f"  {k:8s} {res[k]['auc']:.4f}  落差 {res[k]['auc'] - base:+.4f}")

# 配对差值（逐折，vs full）
print("\n配对差值（逐折，n=240）：")
af = np.array(res["full"]["auc_per_fold"])
for k in ("refonly", "altonly", "delta", "nocpos"):
    ak = np.array(res[k]["auc_per_fold"])
    m = np.isfinite(af) & np.isfinite(ak)
    dd = af[m] - ak[m]; se = dd.std(ddof=1) / np.sqrt(m.sum())
    print(f"  full − {k:8s} = {dd.mean():+.4f} 95%CI[{dd.mean()-1.96*se:+.4f},{dd.mean()+1.96*se:+.4f}]")

json.dump(res, open(f"{OUT}/b3_ablation_v1.json", "w"), indent=2, ensure_ascii=False)
print("\n已写", f"{OUT}/b3_ablation_v1.json")
