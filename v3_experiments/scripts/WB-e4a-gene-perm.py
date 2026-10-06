# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e4a-gene-perm.py  (2026-10-03)
目的：落实 GPT「实验E4：变异配对与基因条件控制」的 A 臂——**基因内标签置换**，
      并顺带把 GPT §七.4 要求的「第一类不确定性（测试基因/变异抽样）」用
      cluster bootstrap 补上。纯 CPU，不需要嵌入。

GPT 原文（E4-A）四条要求，本脚本逐条落实：
  1. 「在训练基因内部置换标签，保留基因阳性比例」
        ⇒ perm_within_gene()：按 gene_symbol 分组，组内 shuffle。
          组内 shuffle 按定义保留该组阳性数 ⇒ 保留基因阳性率。
  2. 「测试标签保持原样」
        ⇒ 只置换训练集；测试集用原始 y。
  3. 「超参数选择不能偷偷使用未置换的验证信息」
        ⇒ fit_select() 只在**置换后的**训练数据上做内部 2 折搜索。
  4. 「该控制主要用于已知基因场景」
        ⇒ 🔴 这是我此前漏掉的前提。当前 240 folds 是**未见基因**（留一基因），
          测试基因根本不在训练集里，置换后模型连"该基因阳性率"都查不到。
          要测"基因相关标签结构足以支持多少性能"，必须在**已知基因**场景做。
        ⇒ 本脚本跑两个场景：
           G2 未见基因 = 冻结的 240 valid folds
           G1 已知基因 = 变异级随机 5 折（同一基因的变异分散到各折）
          两场景对照，才能把"基因身份通道"和"变异特异通道"分开。

为什么置换后仍可能 AUC > 0.5：
  模型可学到「这个基因（经 GC / 密码子使用等序列风格识别）→ 该基因的阳性率」。
  这部分不是变异特异信号，是基因标签结构。置换保留它、破坏后者。

核心量 = retention = (AUC_perm − 0.5) / (AUC_real − 0.5)
  ⇒ 性能中由"基因相关标签结构"支撑的比例。

输出（永不覆盖）：results/supplementary/wb_rerun/wb_frozen/e4a_geneperm_v1.json
"""
import os, json, sys, warnings
import numpy as np
import pandas as pd
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
os.makedirs(OUT, exist_ok=True)

SEED = 42
N_BOOT = 2000            # cluster bootstrap 次数
ONLY = sys.argv[1] if len(sys.argv) > 1 else "all"   # all | g1 | g2 | boot

# ---------------- 1. 数据 ----------------
df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
y = df["label"].values.astype(int)
genes = df["gene_symbol"].values
n = len(df)
print(f"样本 {n}, 基因 {len(pd.unique(genes))}, 阳性率 {y.mean():.4f}")

CENTER = 45
# 数据坑（EXP-006 已记）：ReferenceAllele/AlternateAllele 两列全 'na'，真值在带 VCF 后缀的列
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = df["char_seq"].values

# ---------------- 2. 特征（与 EXP-016 完全一致，含已修的负链双重互补 bug） ----------------
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
cpos = df["cpos"].values.reshape(-1, 1).astype(float) if "cpos" in df.columns else np.zeros((n, 1))
from collections import Counter
cnt = Counter()
for s in seqs:
    for i in range(len(s) - 2):
        cnt[s[i:i + 3]] += 1
TOP_KMER = [k for k, _ in cnt.most_common(16)]
kmer = np.array([[s.count(k) / max(1, len(s)) for k in TOP_KMER] for s in seqs])
B2 = np.hstack([gc.reshape(-1, 1), kmer, cpos])

COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
CIDX = {a + b + c: i for i, (a, b, c) in enumerate(
    (x, y_, z) for x in "ACGT" for y_ in "ACGT" for z in "ACGT")}
CIDX["NNN"] = len(CIDX)
vix_a = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
ph_a = np.load(f"{SUPP}/wb_rerun/e1s_phase_pick.npy")

def _rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s.upper()))

def _onehot(arr):
    M = np.zeros((len(arr), 65))
    for i, c in enumerate(arr):
        M[i, CIDX.get(c, CIDX["AAA"])] = 1.0
    return M

# 🔴🔴 2026-10-03（与 WB-e1-baselines-240folds.py 同步的第三次修）：
#    **char_seq 已经是编码链方向，不做反向互补**；只对负链基因的 ref/alt **碱基**取互补。
#    链向改以 HGVS 的 c. 碱基为基准（0 条不可判）。相位 off = (cpos - 1) % 3。
#    外部基准验证：HGVS 参考氨基酸吻合 99.72%，同义自洽 99.93%。
import re as _re
_PATC = _re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
_hg_ref = []
for _nm in df["Name"]:
    _m = _PATC.search(str(_nm))
    _hg_ref.append(_m.group(2) if _m else None)
strand_plus = np.array([(_hg_ref[i] is not None and _hg_ref[i] == ref_base[i])
                        for i in range(n)])
print(f"链向(以 HGVS 为基准): plus={int(strand_plus.sum())} "
      f"minus={int(n - strand_plus.sum())}")
rc_list, ac_list, ok, anti = [], [], 0, 0
cpos_i = df["cpos"].values.astype(int)
off_arr = (cpos_i - 1) % 3
_b = "TCAG"; _aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
C2A = {}; _i = 0
for _x in _b:
    for _y in _b:
        for _z in _b:
            C2A[_x + _y + _z] = _aa[_i]; _i += 1
COMP_B = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
for gi in range(n):
    s = str(seqs[gi]).upper()          # 不反向互补
    vi = int(vix_a[gi]); off = int(off_arr[gi])
    st = vi - off
    cod = s[st:st + 3]
    if len(cod) != 3 or not (0 <= off < 3):
        rc_list.append("NNN"); ac_list.append("NNN"); continue
    rb = ref_base[gi] if strand_plus[gi] else COMP_B.get(ref_base[gi], "N")
    ab = alt_base[gi] if strand_plus[gi] else COMP_B.get(alt_base[gi], "N")
    if cod[off] != rb:
        rc_list.append("NNN"); ac_list.append("NNN"); continue
    acod = cod[:off] + ab + cod[off + 1:]
    if C2A.get(cod, "X") != C2A.get(acod, "X"):
        anti += 1
    rc_list.append(cod); ac_list.append(acod); ok += 1
B3 = np.hstack([_onehot(rc_list), _onehot(ac_list), gc.reshape(-1, 1), cpos])
print(f"B2 {B2.shape}, B3 {B3.shape} (构造成功 {ok}/{n})")
print(f"T1 氨基酸一致 {ok-anti}/{ok} = {100*(ok-anti)/max(1,ok):.2f}%（不一致 {anti} 条）")

FEATS = {"B2_background": B2, "B3_codon_table": B3}

# ---------------- 3. 探针（E5-A 公平预算：与 EXP-016 同一套） ----------------
def fit_select(Xtr, ytr, seed=SEED):
    """训练集内部 2 折搜索 C；只在**传入的** ytr 上搜索 ⇒ 置换场景下不会偷看原标签。"""
    if len(set(ytr)) < 2:
        return None
    try:
        splits = list(StratifiedKFold(2, shuffle=True, random_state=seed).split(Xtr, ytr))
    except Exception:
        splits = []
    if not splits:
        splits = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    best, best_s = None, -1
    for C in (0.1, 1.0):
        ss = []
        for tr_i, va_i in splits:
            pipe = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=1000, C=C, solver="lbfgs"))
            pipe.fit(Xtr[tr_i], ytr[tr_i])
            pv = pipe.predict_proba(Xtr[va_i])[:, 1]
            ss.append(0.5 if len(set(ytr[va_i])) < 2 else roc_auc_score(ytr[va_i], pv))
        s = float(np.mean(ss))
        if s > best_s:
            best_s, best = s, C
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, C=best, solver="lbfgs"))
    pipe.fit(Xtr, ytr)
    return pipe

# ---------------- 4. 基因内标签置换 ----------------
def perm_within_gene(y_tr, g_tr, rng):
    """组内 shuffle ⇒ 按定义保留每组阳性数（基因阳性率）；单样本组无法破坏。"""
    out = y_tr.copy()
    sizes = pd.Series(g_tr).value_counts()
    multi = 0
    for g in np.unique(g_tr):
        m = g_tr == g
        k = int(m.sum())
        if k > 1:
            idx = np.where(m)[0]
            out[idx] = rng.permutation(y_tr[idx])
            multi += k
    return out, multi, int(sizes.size)

# ---------------- 5. 折定义 ----------------
d = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fold_id = d["fold_id"]; gidx = d["global_index"]
N240 = int(fold_id.max()) + 1

def run_folds(X, folds_te_idx, tr_mask_fn, perm, tag, seed=SEED):
    """folds_te_idx: list of 测试索引数组；tr_mask_fn(te_idx) -> 训练索引数组"""
    rng = np.random.RandomState(seed)
    aucs, pinfo = [], []
    for te in folds_te_idx:
        tr = tr_mask_fn(te)
        ytr = y[tr].copy()
        if perm:
            ytr, multi, ngene = perm_within_gene(ytr, genes[tr], rng)
            # 🔴 2026-10-03 修：原先 append (multi, ngene) 却在汇总时把两者当反了
            #    （mul=sum(ngene)、tot=sum(multi)）⇒ 打印出的 0.176 无意义。
            #    正确：coverage = multi / 训练集样本数。
            pinfo.append((multi, len(ytr)))
        pipe = fit_select(X[tr], ytr, seed)
        if pipe is None:
            aucs.append(np.nan); continue
        s = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s) if len(set(y[te])) > 1 else np.nan)
    aucs = np.array(aucs, dtype=float)
    out = {"auc_per_fold": aucs.tolist(),
           "auc_mean": float(np.nanmean(aucs)),
           "auc_sem": float(np.nanstd(aucs, ddof=1) / np.sqrt(np.isfinite(aucs).sum()))}
    if perm:
        # a = 该折训练集中"所属基因在训练集内有 >=2 条变异"的样本数
        # b = 该折训练集样本总数
        mul = sum(a for a, b in pinfo); tot = sum(b for a, b in pinfo)
        out["train_multi_sample_frac"] = float(mul / max(1, tot))
        out["train_mean_n"] = float(np.mean([b for a, b in pinfo]))
    return out

report = {"meta": {"seed": SEED, "n_boot": N_BOOT, "probe": "LR(C in {0.1,1}, inner 2-fold)"},
          "scenarios": {}, "bootstrap": {}}

# ---------------- 6. G2：未见基因（240 valid folds） ----------------
te_folds_240 = [gidx[fold_id == k] for k in range(N240)]
tr_mask_240 = lambda te: np.setdiff1d(np.arange(n), te)   # 全集取补集（与 EXP-016 一致）

if ONLY in ("all", "g2"):
    print("\n=== G2 未见基因（240 valid folds）===")
    g2 = {}
    for fname, X in FEATS.items():
        for perm in (False, True):
            tag = "perm" if perm else "real"
            r = run_folds(X, te_folds_240, tr_mask_240, perm, tag)
            g2[f"{fname}|{tag}"] = r
            print(f"  {fname:16s} {tag:5s} AUC={r['auc_mean']:.4f} ±{r['auc_sem']:.4f}"
                  + (f"  (训练集多样本基因覆盖 {r['train_multi_sample_frac']:.3f})" if perm else ""))
    # retention
    for fname in FEATS:
        a = g2[f"{fname}|real"]["auc_mean"]; b = g2[f"{fname}|perm"]["auc_mean"]
        g2[f"{fname}|retention"] = float((b - 0.5) / (a - 0.5)) if abs(a - 0.5) > 1e-9 else None
        print(f"  {fname:16s} retention = {g2[f'{fname}|retention']}")
    report["scenarios"]["G2_unseen_gene"] = g2

# ---------------- 7. G1：已知基因（变异级随机 5 折 × 5 次重复） ----------------
if ONLY in ("all", "g1"):
    print("\n=== G1 已知基因（变异级随机 5 折 × 5 次）===")
    g1 = {}
    NREP = 5
    for fname, X in FEATS.items():
        for perm in (False, True):
            tag = "perm" if perm else "real"
            all_auc, seen_frac = [], []
            for rep in range(NREP):
                skf = StratifiedKFold(5, shuffle=True, random_state=SEED + rep)
                tes = [te for _, te in skf.split(np.zeros((n, 1)), y)]
                for te in tes:
                    tr = np.setdiff1d(np.arange(n), te)
                    seen_frac.append(float(np.isin(genes[te], np.unique(genes[tr])).mean()))
                r = run_folds(X, tes, lambda te: np.setdiff1d(np.arange(n), te), perm, tag,
                              seed=SEED + rep)
                all_auc.extend(r["auc_per_fold"])
            arr = np.array(all_auc, dtype=float)
            g1[f"{fname}|{tag}"] = {"auc_per_fold": arr.tolist(),
                                    "auc_mean": float(np.nanmean(arr)),
                                    "auc_sem": float(np.nanstd(arr, ddof=1) / np.sqrt(np.isfinite(arr).sum()))}
            print(f"  {fname:16s} {tag:5s} AUC={np.nanmean(arr):.4f} ±{np.nanstd(arr, ddof=1)/np.sqrt(np.isfinite(arr).sum()):.4f}"
                  f"   (测试基因在训练集中出现的比例 {np.mean(seen_frac):.3f})")
            g1[f"{fname}|seen_gene_frac"] = float(np.mean(seen_frac))
    for fname in FEATS:
        a = g1[f"{fname}|real"]["auc_mean"]; b = g1[f"{fname}|perm"]["auc_mean"]
        g1[f"{fname}|retention"] = float((b - 0.5) / (a - 0.5)) if abs(a - 0.5) > 1e-9 else None
        print(f"  {fname:16s} retention = {g1[f'{fname}|retention']}")
    report["scenarios"]["G1_known_gene"] = g1

# ---------------- 8. cluster bootstrap（第一类不确定性：测试基因抽样） ----------------
if ONLY in ("all", "boot"):
    print(f"\n=== cluster bootstrap（按折=基因重抽样，{N_BOOT} 次）===")
    # 已有逐折预测：CodonBERT C-alt（CD 线，240 folds，LR）
    cb = np.load(f"{SUPP}/logo_cv_predictions_codonbert_task3_synonymous.npz", allow_pickle=True)
    yc = cb["lr_y_true"]; sc = cb["lr_y_proba"]
    cd_auc = np.array([roc_auc_score(yc[fold_id == k], sc[fold_id == k]) for k in range(N240)])

    base = json.load(open(f"{OUT}/baselines_240logo_v1.json"))["results"]
    b2_auc = np.array(base["B2_background|lr"]["auc_per_fold"])
    b3_auc = np.array(base["B3_codon_table|lr"]["auc_per_fold"])

    rng = np.random.RandomState(SEED)
    idx_pool = np.arange(N240)
    boots = {"B2_background": np.zeros(N_BOOT), "B3_codon_table": np.zeros(N_BOOT),
             "CodonBERT_Calt": np.zeros(N_BOOT),
             "B3_minus_CodonBERT": np.zeros(N_BOOT), "B2_minus_CodonBERT": np.zeros(N_BOOT)}
    for b in range(N_BOOT):
        take = rng.choice(idx_pool, N240, replace=True)
        boots["B2_background"][b] = np.nanmean(b2_auc[take])
        boots["B3_codon_table"][b] = np.nanmean(b3_auc[take])
        boots["CodonBERT_Calt"][b] = np.nanmean(cd_auc[take])
        boots["B3_minus_CodonBERT"][b] = np.nanmean(b3_auc[take] - cd_auc[take])
        boots["B2_minus_CodonBERT"][b] = np.nanmean(b2_auc[take] - cd_auc[take])
    print(f"  {'量':24s} {'点估计':>8s} {'boot均值':>8s} {'95%CI':>22s}")
    for k, v in boots.items():
        pt = {"B2_background": b2_auc.mean(), "B3_codon_table": b3_auc.mean(),
              "CodonBERT_Calt": cd_auc.mean(),
              "B3_minus_CodonBERT": (b3_auc - cd_auc).mean(),
              "B2_minus_CodonBERT": (b2_auc - cd_auc).mean()}[k]
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"  {k:24s} {pt:8.4f} {v.mean():8.4f}   [{lo:.4f}, {hi:.4f}]")
        report["bootstrap"][k] = {"point": float(pt), "boot_mean": float(v.mean()),
                                  "ci95": [float(lo), float(hi)],
                                  "boot_se": float(v.std(ddof=1))}
    # 🔴 负例（2026-10-03 修正）：第一版用 rng.permutation(b3_auc) 打散配对——**无效设计**。
    #    mean(a - b) 与元素顺序无关 ⇒ 打散后均值分毫不动（实测 +0.2102 → +0.2102），
    #    区间宽度为 0，什么都测不出来。
    #    配对真正改变的是**方差**，不是均值 ⇒ 负例必须比较两种重抽样的 SE：
    #      配对重抽样：b3 与 cd 用同一组折索引 ⇒ 保留逐基因配对
    #      非配对重抽样：两者各抽各的 ⇒ 配对被破坏
    #    判据：SE_unpaired / SE_paired 明显 > 1 ⇒ 配对确实降低了不确定性。
    npr, nun = [], []
    for _ in range(1000):
        t = rng.choice(idx_pool, N240, replace=True)
        npr.append(np.nanmean(b3_auc[t] - cd_auc[t]))
        t1 = rng.choice(idx_pool, N240, replace=True)
        t2 = rng.choice(idx_pool, N240, replace=True)
        nun.append(np.nanmean(b3_auc[t1] - cd_auc[t2]))
    npr = np.array(npr); nun = np.array(nun)
    ratio = nun.std(ddof=1) / max(1e-12, npr.std(ddof=1))
    print(f"  负例（配对 vs 非配对重抽样）B3−CodonBERT:")
    print(f"    配对   均值 {npr.mean():+.4f}  SE {npr.std(ddof=1):.4f}")
    print(f"    非配对 均值 {nun.mean():+.4f}  SE {nun.std(ddof=1):.4f}")
    print(f"    SE 比值 = {ratio:.2f}  ( >1 说明配对确实降低了不确定性 )")
    report["bootstrap"]["NEGATIVE_paired_vs_unpaired"] = {
        "paired_mean": float(npr.mean()), "paired_se": float(npr.std(ddof=1)),
        "unpaired_mean": float(nun.mean()), "unpaired_se": float(nun.std(ddof=1)),
        "se_ratio": float(ratio),
        "verdict": "PASS 配对有效" if ratio > 1.2 else "FAIL 配对无效（说明逐基因配对没起作用）"}

with open(f"{OUT}/e4a_geneperm_v1.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n已写入", f"{OUT}/e4a_geneperm_v1.json")
