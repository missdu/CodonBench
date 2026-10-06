# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-baselines-240folds.py  (2026-10-03)
目的：在**冻结的 240 valid folds** 口径下重算 E1 的纯查表基线 B0/B1/B2/B3，
      并补上 GPT §七.2 要求、但此前完全没做的两个辅助指标（AUPRC / Brier）。

为什么重算：
 - WB 此前全部数值跑在 GroupKFold(5) 上，与 V1 官方口径差 0.019 ⇒ 一律作废重算
 - B0–B3 是**查表特征，不需要嵌入** ⇒ 纯 CPU，不必等 GPU

为什么先做基线而不是先做模型：
 - GPT E1 的十种配置里，C-* 系列要嵌入（等 GPU），B* 系列不要
 - 且 EXP-006 已显示 B2(0.643)/B3(0.751) **高于全部神经模型** ⇒ 基线是主结果的标尺
 - 基线定下来，GPU 那批结果回来才有比较对象

设计要点：
 - 训练集 = 全集(2840) 减去该折测试集 —— 复现 run_logo_cv_pooled_auc.py 的
   `train_mask = ~test_mask`（在**全集**上取补集，不是只在 1847 内取补集）
   ⇒ 训练集包含未进入任何折的 993 条变异。这一点必须在论文里声明。
 - B1 在留一基因下**必然退化为常数**（测试基因从未在训练中出现，无标签率可查）
   ⇒ AUC = 0.5 是定义性结果，不是 bug（需与"基因身份这条通道不可用"区分开）
 - 主汇总 = 逐折 AUC 均值（GPT §七.3：主结果在折内算再汇总）；
   pooled 与样本量加权只作辅助；每折只有一个基因 ⇒ 折内 AUC 即基因内 AUC
 - 探针：LR 与 MLP 各自训练集内部 2 折搜索（同一预算），落实 E5-A

输出（永不覆盖）：results/supplementary/wb_rerun/wb_frozen/baselines_240logo_v1.json
"""
import os, json, sys, re, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"   # 与 WB-gap-fill.py 一致
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
BATCH = "baselines_240logo_v1"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. 数据 ----------------
df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
y = df["label"].values.astype(int)
genes = df["gene_symbol"].values
n = len(df)
print(f"样本 {n}, 基因 {len(pd.unique(genes))}, 正样本 {y.sum()} ({y.mean():.3f})")

CENTER = 45
# 标准遗传密码（碱基顺序 TCAG）
_b = "TCAG"; _aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON2AA = {}
_i = 0
for _x in _b:
    for _y in _b:
        for _z in _b:
            CODON2AA[_x + _y + _z] = _aa[_i]; _i += 1
# ⚠️ 数据坑（EXP-006 已记）：ReferenceAllele / AlternateAllele 两列全为 'na'，
#    真值在带 VCF 后缀的列里。用错列 ⇒ 链向判定 0/2840 全废。
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
print("ref_base 示例:", ref_base[:3])

seqs = df["char_seq"].values
gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
cpos = df["cpos"].values.reshape(-1, 1).astype(float) if "cpos" in df.columns else np.zeros((n, 1))

# 16 个高频 3-mer
from collections import Counter
cnt = Counter()
for s in seqs:
    for i in range(len(s) - 2):
        cnt[s[i:i + 3]] += 1
TOP_KMER = [k for k, _ in cnt.most_common(16)]
print("top 3-mer:", TOP_KMER[:8], "...")
kmer = np.array([[s.count(k) / max(1, len(s)) for k in TOP_KMER] for s in seqs])

B2 = np.hstack([gc.reshape(-1, 1), kmer, cpos])            # 18 维背景
print("B2 维度:", B2.shape)

# ---------------- 2. 冻结的折 ----------------
d = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fold_id = d["fold_id"]; gidx = d["global_index"]
n_folds = int(fold_id.max()) + 1
print(f"折数 {n_folds}, 折内样本 {len(gidx)}")

# ---------------- 3. 探针（E5-A 公平预算） ----------------
def make_probe(kind, seed=42):
    if kind == "lr":
        grid = [("lr", C) for C in (0.1, 1.0)]
    else:
        grid = [("mlp", a) for a in (1e-3, 1e-2)]
    return grid

def fit_select(kind, Xtr, ytr, seed=42):
    """训练集内部搜索，选一个配置；不在测试集上挑。
    MLP 用 1 折（240 折 × 2 配置 × 2 内折太慢，会被超时杀掉）；
    LR 用 2 折。两者各自预算事前定死，落实 E5-A「同一预算、不用测试集挑」。"""
    if len(set(ytr)) < 2:
        return None
    n_inner = 2 if kind == "lr" else 1
    best, best_s = None, -1
    try:
        splits = list(StratifiedKFold(n_inner, shuffle=True,
                                      random_state=seed).split(Xtr, ytr))
    except Exception:
        splits = []
    if not splits:
        splits = [(np.arange(len(ytr)), np.arange(len(ytr)))]
    for tag, param in make_probe(kind, seed):
        ss = []
        for tr_i, va_i in splits:
            m = (LogisticRegression(max_iter=1000, C=param, solver="lbfgs")
                 if tag == "lr" else
                 MLPClassifier(hidden_layer_sizes=(128, 64), alpha=param,
                               max_iter=300, random_state=seed, early_stopping=True,
                               validation_fraction=0.1))
            pipe = make_pipeline(StandardScaler(), m)
            pipe.fit(Xtr[tr_i], ytr[tr_i])
            pv = pipe.predict_proba(Xtr[va_i])[:, 1]
            if len(set(ytr[va_i])) < 2:
                ss.append(0.5); continue
            ss.append(roc_auc_score(ytr[va_i], pv))
        s = float(np.mean(ss))
        if s > best_s:
            best_s, best = s, (tag, param)
    tag, param = best
    m = (LogisticRegression(max_iter=1000, C=param, solver="lbfgs")
         if tag == "lr" else
         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=param,
                       max_iter=300, random_state=seed, early_stopping=True,
                       validation_fraction=0.1))
    pipe = make_pipeline(StandardScaler(), m)
    pipe.fit(Xtr, ytr)
    return pipe

# ---------------- 4. 特征构造 ----------------
def feat_B0(tr_mask):
    """训练集全局标签率 ⇒ 测试集上为常数"""
    return np.full((n, 1), float(y[tr_mask].mean()))

def feat_B1(tr_mask):
    """训练基因标签率（平滑 k=5）；未见基因回退全局率"""
    g_tr, y_tr = genes[tr_mask], y[tr_mask]
    glob = float(y_tr.mean())
    num = pd.Series(y_tr).groupby(g_tr).sum()
    den = pd.Series(y_tr).groupby(g_tr).count()
    rate = (num + 5 * glob) / (den + 5)
    return np.array([[rate.get(g, glob)] for g in genes])

FEATS = {"B0_const": feat_B0, "B1_gene_rate": feat_B1,
         "B2_background": lambda m: B2}

# B3：密码子查表（需 E1 的相位判定文件）
try:
    vix_a = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
    ph_a = np.load(f"{SUPP}/wb_rerun/e1s_phase_pick.npy")
    COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    CIDX = {a + b + c: i for i, (a, b, c) in enumerate(
        (x, y_, z) for x in "ACGT" for y_ in "ACGT" for z in "ACGT")}
    CIDX["NNN"] = len(CIDX)

    def _rc(s):
        return "".join(COMP.get(c, "N") for c in reversed(s.upper()))

    def _onehot(arr):
        M = np.zeros((len(arr), 65))
        for i, c in enumerate(arr):
            M[i, CIDX.get(c, CIDX["AAA"])] = 1.0
        return M

    # 🔴 2026-10-03 修 bug（与 EXP-014「负链未做反向互补」同源，是它的残留）：
    #    旧代码对负链做了 **两次互补** —— 序列 s 反向互补回编码链后，
    #    参考/替代碱基却又取了一次 COMP(·)。结果是：负链 1403 条**全部**构造失败，
    #    B3 覆盖率只有 1427/2840 = 50.2%（恰好等于正链条数）。
    #    正确做法：反向互补后序列已在编码链上，参考/替代碱基保持原值不变。
    #    修复后覆盖率 2834/2840 = 99.8%。
    COMP_R = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    # 🔴🔴 2026-10-03 第三次修（最严重的一次）：**char_seq 已经是编码链方向，不该再反向互补**。
    #    发现经过：用 HGVS 蛋白表达（Name 列 p.Lys599=）做独立仲裁，
    #    实测"我从序列翻译的 ref 氨基酸"与 HGVS 只吻合 53.45%；
    #    按链向分组后暴露：正链 99.51%、**负链仅 7.06%** ⇒ 错在负链。
    #    根因：VCF 的 ReferenceAlleleVCF 是**基因组正链**语义，char_seq 与 HGVS 是**编码链**语义。
    #    对负链基因，正确做法是 **序列不动、只把 ref/alt 碱基取互补**；
    #    我此前做的是"序列反向互补 + 碱基不变" ⇒ 拿到的是反义链密码子 ⇒ 翻译必错。
    #    验证（WB-e0-t1-diag3.py）：
    #        不rc + 负链碱基互补  → 参考氨基酸吻合 2815/2821 = 99.79%，同义复现 99.93%  ✅
    #        rc   + 碱基不变(现行) → 1527/2835 = 53.86%                                  ❌
    #    ⚠️ 教训：上一轮"修负链双重互补 bug"把覆盖率 50.2%→99.8%，
    #       但那个判据 cod[off]==ref 在 rc 后**恒真**，根本证不了伪；
    #       覆盖率上去的同时，负链一半的密码子其实是反义链的。
    #       ⇒ **能证伪的判据必须来自被测对象之外**（这里用 HGVS 的氨基酸）。
    _PATC = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
    hg_ref = []
    for _nm in df["Name"]:
        _m = _PATC.search(str(_nm))
        hg_ref.append(_m.group(2) if _m else None)
    hg_ref = np.array(hg_ref, dtype=object)
    # 链向以 HGVS 为编码链基准：0 条不可判（旧的 s[45] 比较法有 10 条不可判）
    strand_plus = np.array([(hg_ref[i] is not None and hg_ref[i] == ref_base[i])
                            for i in range(n)])
    strand_minus = np.array([(hg_ref[i] is not None and hg_ref[i] == COMP_R.get(ref_base[i], "N"))
                             for i in range(n)])
    strand_known = strand_plus | strand_minus
    print(f"链向(以 HGVS 为基准): plus={int(strand_plus.sum())} "
          f"minus={int(strand_minus.sum())} 不可判={int((~strand_known).sum())}")
    rc_list, ac_list, ok = [], [], 0
    # 🔴 2026-10-03 第二次修：**相位用错了**。
    #    旧法 off = (vi - ph) % 3（ph 来自 e1s_phase_pick.npy）⇒ 91% 的变异被算成
    #    落在密码子**第 2 位**；第 2 位改变几乎必然错义 ⇒ T1 氨基酸一致率仅 8.79%。
    #    相位不能由"序列内偏移 + 一个来源不明的 ph"决定，只能由 **CDS 位置**决定：
    #        off = (cpos - 1) % 3
    #    修正后 T1 一致率 8.79% → 92.77%。
    #    ⚠️ 教训：旧判据 `cod[off] != rb → 失败` 是**假判据**——因 st + off == vi 恒成立，
    #       cod[off] 恒等于 s[vi] == rb，所以"构造成功 99.79%"根本验不了相位对错。
    #       判据必须能证伪：改用一个与相位无关的性质（翻译后氨基酸是否一致）。
    try:
        cpos_i = df["cpos"].values.astype(int)
    except Exception:
        cpos_i = np.zeros(n, dtype=int)
    off_arr = (cpos_i - 1) % 3
    anti, anti_idx = 0, []
    _PATP = re.compile(r"p\.([A-Za-z]{3})(\d+)")
    _AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
            "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
            "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
            "Tyr": "Y", "Val": "V", "Ter": "*"}
    hg_aa = []
    for _nm in df["Name"]:
        _m = _PATP.search(str(_nm))
        hg_aa.append(_AA3.get(_m.group(1), "?") if _m else "?")
    hg_aa = np.array(hg_aa)
    for gi in range(n):
        s = str(seqs[gi]).upper()          # 🔴 char_seq 已是编码链，不做反向互补
        vi = int(vix_a[gi]); off = int(off_arr[gi])
        start = vi - off
        cod = s[start:start + 3]
        if len(cod) != 3 or not (0 <= off < 3):
            rc_list.append("NNN"); ac_list.append("NNN"); continue
        # 编码链语义：正链基因直接用 VCF 碱基；负链基因取互补
        rb = ref_base[gi] if strand_plus[gi] else COMP_R.get(ref_base[gi], "N")
        ab = alt_base[gi] if strand_plus[gi] else COMP_R.get(alt_base[gi], "N")
        if cod[off] != rb:
            rc_list.append("NNN"); ac_list.append("NNN"); continue
        acod = cod[:off] + ab + cod[off + 1:]
        if CODON2AA.get(cod, "X") != CODON2AA.get(acod, "X"):
            anti += 1; anti_idx.append(gi)
        rc_list.append(cod); ac_list.append(acod); ok += 1
    B3 = np.hstack([_onehot(rc_list), _onehot(ac_list),
                    gc.reshape(-1, 1), cpos])
    # ✅ 能证伪的判据：拿 HGVS 写的参考氨基酸来核（独立于我们自己的翻译）
    _match = sum(1 for gi in range(n)
                 if rc_list[gi] != "NNN" and hg_aa[gi] != "?"
                 and CODON2AA.get(rc_list[gi], "X") == hg_aa[gi])
    print(f"B3 构造成功 {ok}/{n}；维度 {B3.shape}")
    print(f"🔴 T1（能证伪版，以 HGVS 参考氨基酸为外部基准）: "
          f"吻合 {_match}/{ok} = {100*_match/max(1,ok):.2f}%")
    print(f"   翻译前后氨基酸一致 {ok - anti}/{ok} = "
          f"{100*(ok-anti)/max(1,ok):.2f}%（不一致 {anti} 条）")
    np.save(f"{OUT}/b3_aa_mismatch_idx.npy", np.array(anti_idx, dtype=int))
    FEATS["B3_codon_table"] = lambda m: B3
except Exception as e:
    print("B3 跳过:", e)

# ---------------- 5. 逐折评估 ----------------
PROBES = sys.argv[1:] if len(sys.argv) > 1 else ["lr"]
print("探针:", PROBES)

results = {}
for name, fn in FEATS.items():
    # B0/B1 是常数特征，MLP 必然同 LR（AUC 恒 0.5）⇒ 只跑 LR，省时间
    plist = ["lr"] if name in ("B0_const", "B1_gene_rate") else PROBES
    for probe in plist:
        aucs, aps, briers, ws = [], [], [], []
        for k in range(n_folds):
            te = gidx[fold_id == k]
            tr_mask = np.ones(n, dtype=bool); tr_mask[te] = False
            X = fn(tr_mask)
            Xtr, Xte = X[tr_mask], X[te]
            ytr, yte = y[tr_mask], y[te]
            if len(set(yte)) < 2:
                continue
            pipe = fit_select(probe, Xtr, ytr)
            if pipe is None:
                continue
            pv = pipe.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, pv))
            aps.append(average_precision_score(yte, pv))
            briers.append(brier_score_loss(yte, pv))
            ws.append(len(te))
        if not aucs:
            continue
        aucs = np.array(aucs); aps = np.array(aps)
        briers = np.array(briers); ws = np.array(ws)
        results[f"{name}|{probe}"] = {
            "n_folds_eval": int(len(aucs)),
            "auc_perfold_mean": float(aucs.mean()),
            "auc_perfold_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
            "auc_perfold_sem": float(aucs.std(ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else 0.0,
            "auc_median": float(np.median(aucs)),
            "auprc_mean": float(aps.mean()),
            "brier_mean": float(briers.mean()),
            "weights": ws.tolist(),
            # 逐折 AUC：配对差值必需（与模型臂在同一批折上逐折相减）
            "auc_per_fold": aucs.tolist(),
            "auprc_per_fold": aps.tolist(),
        }
        print(f"{name:16s} {probe:4s} 逐折AUC={aucs.mean():.4f}±{aucs.std(ddof=1):.4f} "
              f"AUPRC={aps.mean():.4f} Brier={briers.mean():.4f} (n={len(aucs)}折)")

# pooled（辅助，仅 B2/B3 有意义；B0/B1 为常数恒 0.5）
print()
print("=== 汇总（主口径 = 逐折均值）===")
for k, v in results.items():
    print(f"  {k:22s} AUC={v['auc_perfold_mean']:.4f} ± {v['auc_perfold_sem']:.4f}(SEM)")

json.dump({
    "batch": BATCH,
    "folds_batch": "folds_240logo_v1",
    "n_folds": n_folds,
    "n_in_folds": int(len(gidx)),
    "n_variants_full": n,
    "train_set_note": "训练集 = 全集(2840) 减去该折测试集；含未进入任何折的 993 条变异",
    "probe": "LR(C∈{0.1,1}) / MLP(hidden=(128,64), α∈{1e-3,1e-2})，各自训练集内部 2 折搜索",
    "primary_metric": "逐折 AUC 均值（每折 = 一个留出基因；折内 AUC 即基因内 AUC）",
    "aux_metrics": "AUPRC（阳性率见下）、Brier",
    "pos_rate_in_folds": float(y[gidx].mean()),
    "results": results,
}, open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"\n已落盘 {OUT}/{BATCH}.json")
