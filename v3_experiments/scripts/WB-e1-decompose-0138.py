#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
分解结论三的 +0.138：这个差值由哪些成分构成（纯 CPU，不再占用 GPU）。

背景
----
结论三的主结果是一个**差值**：

    Δ = AUC(响应向量 alt−ref) − AUC(位置匹配对照 pse−ref) = +0.138

它只回答"存在且可迁移"，不回答"由什么构成"。审稿人会问：
    这 0.138 是模型读懂了"哪个碱基变成哪个碱基"，
    还是只是"模型动得更厉害"，或"对照组里有三分之一根本没动"？

本脚本用已落盘的三臂嵌入把 Δ 拆开，并在**两种样本口径**下各做一遍：

  口径 A（全样本）      n=1839，与主文 Table S42 完全一致
  口径 B（对照已构造）  只保留对照序列确实与参考序列不同的样本

口径 B 不是可选项。位置匹配对照要求"同义、同位、且不等于 ref/alt 的第三个密码子"；
只有两个密码子的氨基酸（Phe/Tyr/His/Gln/Asn/Lys/Asp/Glu/Cys 等）在 ref 与 alt 已占满
家族时构造不出第三个密码子，构造脚本按约定原样返回参考序列，于是该样本的对照响应向量
是**零向量**。零向量在 pooled AUC 里表现为并列，会把对照臂整体压向 0.5，
从而把 Δ 抬高。口径 B 是排除这种稀释后的估计。

分解的四个切面（两种口径下各做一遍）
------------------------------------
A. 载体分解      Δ 拆成"幅度 ‖Δ‖"（动了多少）与"方向 Δ/‖Δ‖"（往哪动）
B. 廉价特征分解  把"不读模型就能算的序列特征"回归掉，看残余还剩多少
C. 分层分解      按密码子使用频率变化的符号分层，看是否只由一层驱动
D. 秩分解        Δ 的 PCA 谱，看信息集中在几个维度

两项诊断（本身即 0.138 的可能来源）
----------------------------------
E. 对照覆盖率    对照序列与参考序列相同的样本有多少（口径 A/B 之差）
F. 构成偏差      真实臂与对照臂在廉价特征上的分布是否可比

口径
----
与 WB-e1-psecenter-extract.py 一致：LOGO-240 逐折训练、pooled OOF AUC、
基因聚类配对自举（1,000 次）。缺任何必需输入即报错退出，不静默。

用法
----
python WB-e1-decompose-0138.py --model cdsbert [--pca] [--mlp] [--boot 1000]
"""
import os, sys, json, argparse, time
os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results/supplementary/wb_rerun"
FRZ = OUT / "wb_frozen"
CDS_JSON = BASE / "data/task2_clinvar/cds_sequences.json"

# ---------- 遗传密码 ----------
_B = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_LIST = [b1 + b2 + b3 for b1 in _B for b2 in _B for b3 in _B]
AA_OF = {c: _AAS[i] for i, c in enumerate(CODON_LIST)}
IDX_OF = {c: i for i, c in enumerate(CODON_LIST)}
FAMILY = {}
for c, a in AA_OF.items():
    if a != "*":
        FAMILY.setdefault(a, []).append(c)
BASES = "ACGT"
N_CHEAP = 8 + 16          # 序列类廉价特征：8 连续 + 16 碱基变化 one-hot


def die(m):
    print(f"FAILED: {m}", flush=True)
    sys.exit(2)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logo240_oof(X, y, fold_id, mlp=False, seed=42):
    """逐折留出，返回折外分数。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    oof = np.full(len(y), np.nan)
    n_fold_used = 0
    for f in np.unique(fold_id):
        te = fold_id == f
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        if mlp:
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                early_stopping=True, random_state=seed)
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        n_fold_used += 1
    return oof, n_fold_used


def build_codon_freq():
    """从数据集自身 CDS 统计同义家族内相对密码子频率（无外部依赖）。"""
    if not CDS_JSON.exists():
        die(f"缺 CDS 表 {CDS_JSON}")
    d = json.load(open(CDS_JSON))
    lut = np.full(256, 255, dtype=np.uint8)
    for i, b in enumerate("TCAG"):
        lut[ord(b)] = i
        lut[ord(b.lower())] = i
    cnt = np.zeros(64, dtype=np.int64)
    n_tx = 0
    for k in d:
        a = np.frombuffer(d[k].encode("ascii", "ignore"), dtype=np.uint8)
        a = a[:len(a) - len(a) % 3]
        if len(a) < 3:
            continue
        m = a.reshape(-1, 3)
        m = m[(lut[m] != 255).all(1)]
        if len(m) == 0:
            continue
        idx = lut[m[:, 0]].astype(np.int64) * 16 + lut[m[:, 1]] * 4 + lut[m[:, 2]]
        cnt += np.bincount(idx, minlength=64)
        n_tx += 1
    freq = np.zeros(64); rank = np.zeros(64)
    for aa, fam in FAMILY.items():
        ii = [IDX_OF[c] for c in fam]
        tot = cnt[ii].sum()
        if tot == 0:
            continue
        for i in ii:
            freq[i] = cnt[i] / tot
        for r, i in enumerate(sorted(ii, key=lambda i: -freq[i])):
            rank[i] = r + 1
    return {"count": cnt, "freq": freq, "rank": rank,
            "n_tx": n_tx, "total_codons": int(cnt.sum())}


def cheap_features(seq_arm, ref_seq, vpos, off, cf, ebd, cpos, cds_len):
    """对一条臂算廉价特征（不读模型）。返回 (24+2 维矩阵, 诊断量)。"""
    n = len(seq_arm)
    F = np.zeros((n, N_CHEAP), dtype=np.float64)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        s = str(seq_arm[i]); r = str(ref_seq[i])
        st = vpos[i] - off[i]
        if st < 0 or st + 3 > len(s) or st + 3 > len(r):
            continue
        c_arm = s[st:st + 3].upper(); c_ref = r[st:st + 3].upper()
        d = [j for j in range(3) if c_ref[j] != c_arm[j]]
        if len(d) != 1:
            continue
        j = d[0]
        b_ref, b_alt = c_ref[j], c_arm[j]
        if b_ref not in BASES or b_alt not in BASES:
            continue
        ia, ib = IDX_OF.get(c_arm, -1), IDX_OF.get(c_ref, -1)
        f_arm = cf["freq"][ia] if ia >= 0 else 0.0
        f_ref = cf["freq"][ib] if ib >= 0 else 0.0
        r_arm = cf["rank"][ia] if ia >= 0 else 0.0
        r_ref = cf["rank"][ib] if ib >= 0 else 0.0
        fam_sz = len(FAMILY.get(AA_OF.get(c_ref, "*"), []))
        gc = (1 if b_alt in "GC" else 0) - (1 if b_ref in "GC" else 0)
        pur = lambda b: 1 if b in "AG" else 0
        transi = 1 if pur(b_ref) == pur(b_alt) else 0
        row = [f_arm - f_ref, abs(f_arm - f_ref), float(gc), float(transi),
               r_ref - r_arm, float(fam_sz), f_arm, f_ref]
        oh = [0.0] * 16
        if b_ref != b_alt:
            oh[BASES.index(b_ref) * 4 + BASES.index(b_alt)] = 1.0
        F[i] = np.array(row + oh)
        ok[i] = True
    pos = np.column_stack([np.asarray(ebd, float),
                           np.asarray(cpos, float) / np.maximum(np.asarray(cds_len, float), 1)])
    return np.column_stack([F, pos]), {"delta_freq": F[:, 0],
                                       "abs_delta_freq": F[:, 1], "ok": ok}


def residualize(X, C, fold_id):
    """逐折把 X 对 C 做线性回归取残差（标准化也在折内 fit）。"""
    X = np.asarray(X, dtype=np.float64); C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    R = np.zeros_like(X)
    for f in np.unique(fold_id):
        te = fold_id == f; tr = ~te
        sc = StandardScaler().fit(C[tr])
        Ctr, Cte = sc.transform(C[tr]), sc.transform(C[te])
        if Ctr.shape[1] == 0:
            R[tr] = X[tr]; R[te] = X[te]; continue
        reg = LinearRegression().fit(Ctr, X[tr])
        R[tr] = X[tr] - reg.predict(Ctr)
        R[te] = X[te] - reg.predict(Cte)
    return R


def paired_boot(y, oa, ob, genes, n_boot=1000, seed=11):
    ug = np.unique(genes)
    gi = {g: np.where(genes == g)[0] for g in ug}
    rng = np.random.RandomState(seed)
    d0 = auroc(y, oa) - auroc(y, ob)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            bs.append(auroc(y[idx], oa[idx]) - auroc(y[idx], ob[idx]))
        except Exception:
            pass
    if not bs:
        return None
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(d0), 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": len(bs), "cluster_unit": "gene"}


def run_scope(tag, m, D_real, D_ctrl, D_p15, C_real, C_ctrl, y, fold_id,
              genes, a, cf=None, info_real=None, info_ctrl=None):
    """在一种样本口径下跑完整分解。m 为布尔掩码（相对全样本）。"""
    print(f"\n{'='*66}\n口径 {tag}  n={int(m.sum())}\n{'='*66}", flush=True)
    Dr, Dc, Dp = D_real[m], D_ctrl[m], D_p15[m]
    Cr, Cc = C_real[m], C_ctrl[m]
    yy, ff, gg = y[m], fold_id[m], genes[m]

    nrm_r = np.linalg.norm(Dr, axis=1)
    nrm_c = np.linalg.norm(Dc, axis=1)
    eps = 1e-12
    arms = {
        "full_real": Dr, "full_ctrl": Dc, "full_p15": Dp,
        "norm_real": nrm_r.reshape(-1, 1), "norm_ctrl": nrm_c.reshape(-1, 1),
        "dir_real": Dr / np.maximum(nrm_r, eps)[:, None],
        "dir_ctrl": Dc / np.maximum(nrm_c, eps)[:, None],
        "cheap_real": Cr, "cheap_ctrl": Cc,
    }
    arms["resid_real"] = residualize(Dr, np.column_stack([nrm_r.reshape(-1, 1), Cr]), ff)
    arms["resid_ctrl"] = residualize(Dc, np.column_stack([nrm_c.reshape(-1, 1), Cc]), ff)

    oofs, nf = {}, {}
    for k, X in arms.items():
        oofs[k], nf[k] = logo240_oof(X, yy, ff)
        print(f"  {k:12s} AUC={auroc(yy, oofs[k]):.4f}  (折 {nf[k]})", flush=True)

    out = {"n": int(m.sum()), "n_folds_used": int(nf["full_real"]),
           "auc": {k: round(float(auroc(yy, v)), 4) for k, v in oofs.items()}}

    pairs = [("full_real", "full_ctrl", "总差值（结论三主数）"),
             ("norm_real", "norm_ctrl", "仅幅度 ‖Δ‖"),
             ("dir_real", "dir_ctrl", "仅方向 Δ/‖Δ‖"),
             ("resid_real", "resid_ctrl", "残差（扣掉幅度与廉价特征）"),
             ("full_real", "full_p15", "总差值（对上游对照）")]
    out["contrasts"] = {}
    print(f"  --- 关键对比（{tag}）---", flush=True)
    for a1, a2, lab in pairs:
        pb = paired_boot(yy, oofs[a1], oofs[a2], gg, n_boot=a.boot)
        out["contrasts"][f"{a1}_vs_{a2}"] = {"label": lab, **pb}
        print(f"    {lab:22s} Δ={pb['delta']:+.4f}  CI [{pb['ci_lo']}, {pb['ci_hi']}]  "
              f"排除零={pb['excludes_zero']}", flush=True)

    # 切面 C：分层
    dfr = info_real["delta_freq"][m]
    strata = {"频率下降": dfr < -0.02, "频率上升": dfr > 0.02,
              "近似不变": np.abs(dfr) <= 0.02}
    out["strata"] = {}
    print(f"  --- 分层（按真实臂密码子频率变化，{tag}）---", flush=True)
    for lab, sm in strata.items():
        if sm.sum() < 50 or len(np.unique(yy[sm])) < 2:
            out["strata"][lab] = {"n": int(sm.sum()), "note": "样本不足，未评估"}
            print(f"    {lab}: n={int(sm.sum())} 样本不足", flush=True)
            continue
        orr, ocr = oofs["full_real"][sm], oofs["full_ctrl"][sm]
        pb = paired_boot(yy[sm], orr, ocr, gg[sm], n_boot=a.boot)
        out["strata"][lab] = {"n": int(sm.sum()),
                              "auc_real": round(float(auroc(yy[sm], orr)), 4),
                              "auc_ctrl": round(float(auroc(yy[sm], ocr)), 4),
                              "delta": pb["delta"], "ci_lo": pb["ci_lo"],
                              "ci_hi": pb["ci_hi"], "excludes_zero": pb["excludes_zero"]}
        print(f"    {lab:8s} n={int(sm.sum()):4d}  real={auroc(yy[sm],orr):.4f}  "
              f"ctrl={auroc(yy[sm],ocr):.4f}  Δ={pb['delta']:+.4f}  "
              f"CI [{pb['ci_lo']}, {pb['ci_hi']}]", flush=True)

    # 负对照
    out["negative_control"] = {}
    for sd in [7, 2026]:
        rng = np.random.RandomState(sd)
        ys = yy.copy(); rng.shuffle(ys)
        o, _ = logo240_oof(Dr, ys, ff)
        out["negative_control"][f"seed{sd}"] = round(float(auroc(ys, o)), 4)
    print(f"  --- 标签打乱负对照（{tag}）: {out['negative_control']}", flush=True)

    if a.pca:
        Dc2 = Dr - Dr.mean(0, keepdims=True)
        Z = PCA().fit_transform(Dc2)
        out["rank"] = {}
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            if k > Z.shape[1]:
                continue
            o, _ = logo240_oof(Z[:, :k], yy, ff)
            out["rank"][str(k)] = round(float(auroc(yy, o)), 4)
        print(f"  --- 秩分解（{tag}）: {out['rank']}", flush=True)

    if a.mlp:
        o_r, _ = logo240_oof(Dr, yy, ff, mlp=True)
        o_c, _ = logo240_oof(Dc, yy, ff, mlp=True)
        pb = paired_boot(yy, o_r, o_c, gg, n_boot=a.boot)
        out["mlp"] = {"auc_real": round(float(auroc(yy, o_r)), 4),
                      "auc_ctrl": round(float(auroc(yy, o_c)), 4), **pb}
        print(f"  --- 非线性读出器（{tag}）: real={out['mlp']['auc_real']:.4f}  "
              f"ctrl={out['mlp']['auc_ctrl']:.4f}  Δ={pb['delta']:+.4f}  "
              f"CI [{pb['ci_lo']}, {pb['ci_hi']}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--pca", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    print(f"=== 分解 +0.138 | {a.model} | 纯 CPU ===", flush=True)

    p = OUT / f"e1_psecenter_emb_{a.model}.npz"
    if not p.exists():
        die(f"缺嵌入 {p}")
    z = np.load(p, allow_pickle=True)
    E_ref, E_alt, E_pse, E_p15 = z["ref"], z["alt"], z["psecenter"], z["pse15"]
    y = np.asarray(z["labels"], int)
    fold_id = np.asarray(z["fold_id"], int)
    gi = np.asarray(z["global_index"], int)
    D_real = E_alt - E_ref; D_ctrl = E_pse - E_ref; D_p15 = E_p15 - E_ref
    n = len(y)
    print(f"  n={n}  折数={len(np.unique(fold_id))}  维度={D_real.shape[1]}", flush=True)

    gp = OUT / "e1s_genes.npy"
    if not gp.exists():
        die(f"缺基因标签 {gp}")
    fx0 = pd.read_csv(FRZ / "dataset_v2" / "variants.tsv", sep="\t")
    genes_all = np.load(gp, allow_pickle=True)
    if len(genes_all) != len(fx0):
        die(f"基因标签 {len(genes_all)} vs 冻结包 {len(fx0)}")
    genes = np.asarray(genes_all)[gi]

    # ---------- 诊断 E ----------
    nrm_r = np.linalg.norm(D_real, axis=1)
    nrm_c = np.linalg.norm(D_ctrl, axis=1)
    zero_c = nrm_c < 1e-12
    print(f"\n[诊断 E] 对照臂零向量（对照==参考，未构造成功）: "
          f"{int(zero_c.sum())}/{n} ({zero_c.mean()*100:.2f}%)", flush=True)
    print(f"         真实臂零向量: {int((nrm_r < 1e-12).sum())}", flush=True)
    print(f"         ‖Δ‖ 中位数  真实 {np.median(nrm_r):.4f} | 对照 {np.median(nrm_c):.4f} "
          f"| 对照(已构造) {np.median(nrm_c[~zero_c]):.4f}", flush=True)
    print(f"         致病比例  全样本 {y.mean():.3f} | 已构造对照子集 {y[~zero_c].mean():.3f} "
          f"| 零向量子集 {y[zero_c].mean():.3f}", flush=True)

    # 序列与相位
    S_ref = np.load(OUT / "e1s_ref.npy", allow_pickle=True)
    S_alt = np.load(OUT / "e1s_alt.npy", allow_pickle=True)
    S_pse = np.load(OUT / "e1s_psecenter.npy", allow_pickle=True)
    cpos_all = pd.to_numeric(fx0["cds_position"], errors="coerce").values
    off_all = np.where(np.isnan(cpos_all), -1, ((cpos_all - 1) % 3).astype(int))
    off = off_all[gi]
    if (off < 0).any():
        die(f"{int((off<0).sum())} 条无 CDS 位置")
    vpos = np.full(n, -1, dtype=int)
    for i, g in enumerate(gi):
        d = [j for j, (x, yy) in enumerate(zip(str(S_ref[g]), str(S_alt[g]))) if x != yy]
        if len(d) == 1:
            vpos[i] = d[0]
    if (vpos < 0).any():
        die(f"{int((vpos<0).sum())} 条 ref/alt 差异位不唯一")

    cf = build_codon_freq()
    print(f"  密码子表: {cf['n_tx']} 转录本 / {cf['total_codons']:,} 密码子", flush=True)

    fx = fx0.iloc[gi]
    ebd = pd.to_numeric(fx["exon_boundary_distance"], errors="coerce").fillna(0).values
    seq_len_col = fx0["seq_len"] if "seq_len" in fx0.columns else pd.Series(np.zeros(len(fx0)))
    cds_len = pd.to_numeric(seq_len_col, errors="coerce").fillna(0).values[gi]

    C_real, info_real = cheap_features([str(S_alt[g]) for g in gi],
                                       [str(S_ref[g]) for g in gi],
                                       vpos, off, cf, ebd, cpos_all[gi], cds_len)
    C_ctrl, info_ctrl = cheap_features([str(S_pse[g]) for g in gi],
                                       [str(S_ref[g]) for g in gi],
                                       vpos, off, cf, ebd, cpos_all[gi], cds_len)
    print(f"  廉价特征维度: {C_real.shape[1]}  "
          f"(成功解析 真实 {int(info_real['ok'].sum())} / 对照 {int(info_ctrl['ok'].sum())})",
          flush=True)

    # ---------- 诊断 F ----------
    print("\n[诊断 F] 真实臂 vs 对照臂的廉价特征分布", flush=True)
    okc = ~zero_c
    print(f"  Δ密码子频率 均值   真实 {info_real['delta_freq'].mean():+.4f} | "
          f"对照(全) {info_ctrl['delta_freq'].mean():+.4f} | "
          f"对照(已构造) {info_ctrl['delta_freq'][okc].mean():+.4f}", flush=True)
    print(f"  |Δ密码子频率| 均值  真实 {info_real['abs_delta_freq'].mean():.4f} | "
          f"对照(已构造) {info_ctrl['abs_delta_freq'][okc].mean():.4f}", flush=True)
    print(f"  GC 变化 均值       真实 {C_real[:,2].mean():+.4f} | "
          f"对照(已构造) {C_ctrl[okc,2].mean():+.4f}", flush=True)
    print(f"  转换比例           真实 {C_real[:,3].mean():.4f} | "
          f"对照(已构造) {C_ctrl[okc,3].mean():.4f}", flush=True)

    res = {
        "model": a.model, "n_total": int(n),
        "protocol": "LOGO-240 逐折训练 / pooled OOF AUC / 基因聚类配对自举",
        "diagnostics": {
            "n_control_zero_vector": int(zero_c.sum()),
            "frac_control_zero_vector": round(float(zero_c.mean()), 4),
            "n_real_zero_vector": int((nrm_r < 1e-12).sum()),
            "norm_median_real": round(float(np.median(nrm_r)), 5),
            "norm_median_control_all": round(float(np.median(nrm_c)), 5),
            "norm_median_control_constructed": round(float(np.median(nrm_c[~zero_c])), 5),
            "pathogenic_fraction_all": round(float(y.mean()), 4),
            "pathogenic_fraction_constructed": round(float(y[~zero_c].mean()), 4),
            "pathogenic_fraction_zero": round(float(y[zero_c].mean()), 4),
            "n_genes_all": int(len(np.unique(genes))),
            "n_genes_constructed": int(len(np.unique(genes[~zero_c]))),
            "mean_delta_codonfreq_real": round(float(info_real["delta_freq"].mean()), 5),
            "mean_delta_codonfreq_ctrl_all": round(float(info_ctrl["delta_freq"].mean()), 5),
            "mean_delta_codonfreq_ctrl_constructed": round(float(info_ctrl["delta_freq"][okc].mean()), 5),
            "mean_dGC_real": round(float(C_real[:, 2].mean()), 5),
            "mean_dGC_ctrl_constructed": round(float(C_ctrl[okc, 2].mean()), 5),
            "mean_transition_real": round(float(C_real[:, 3].mean()), 5),
            "mean_transition_ctrl_constructed": round(float(C_ctrl[okc, 3].mean()), 5),
            "codon_table": {"n_tx": cf["n_tx"], "total_codons": cf["total_codons"]},
        },
    }

    res["scope_all"] = run_scope("A 全样本", np.ones(n, bool), D_real, D_ctrl, D_p15,
                                 C_real, C_ctrl, y, fold_id, genes, a,
                                 info_real=info_real, info_ctrl=info_ctrl)
    res["scope_control_constructed"] = run_scope(
        "B 对照已构造", ~zero_c, D_real, D_ctrl, D_p15, C_real, C_ctrl,
        y, fold_id, genes, a, info_real=info_real, info_ctrl=info_ctrl)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    op = OUT / f"e1_decompose_{a.model}.json"
    json.dump(res, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n写入 {op}  用时 {res['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
