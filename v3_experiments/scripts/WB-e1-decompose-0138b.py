#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
分解 +0.138 的第二式：增量分解（increment），替代残差化。

为什么不用残差化
----------------
第一式用"把廉价特征回归掉取残差"，得到的残余 AUC 掉到 0.37（低于随机），
说明线性残差化破坏了特征几何，那个数不能解释为"剩下的成分"。

本脚本改用增量：在某个基线特征集之上**加入**响应向量，看 AUC 涨多少。
增量是审稿人熟悉的语言——"在已知 X 的前提下，Y 还多解释了多少"。

四层基线，逐层加严
------------------
  L0  无基线                 → 结论三原版的 Δ（+0.138）
  L1  + 位置（外显子接点距离） → 扣除位置后还剩多少
  L2  + 位置 + 廉价序列特征   → 再扣除密码子频率/GC/碱基变化类型后还剩多少
  L3  + 位置 + 廉价 + 幅度    → 连"模型动了多少"也扣掉后还剩多少

每层都报：真实臂增量、对照臂增量、以及二者之差（这一层的核心数）。
两个模型、两种样本口径各做一遍。

用法
----
python WB-e1-decompose-0138b.py --model cdsbert
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results/supplementary/wb_rerun"
FRZ = OUT / "wb_frozen"
CDS_JSON = BASE / "data/task2_clinvar/cds_sequences.json"

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


def die(m):
    print(f"FAILED: {m}", flush=True); sys.exit(2)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    ok = ~np.isnan(s); y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def oof(X, y, fold_id):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    s = np.full(len(y), np.nan)
    for f in np.unique(fold_id):
        te = fold_id == f; tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        s[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return s


def boot_dd(y, af, ab, bf, bb, genes, n_boot=1000, seed=11):
    """二重差 (auc(af)-auc(ab)) - (auc(bf)-auc(bb)) 的基因聚类自举。"""
    ug = np.unique(genes)
    gi = {g: np.where(genes == g)[0] for g in ug}
    rng = np.random.RandomState(seed)
    d0 = (auroc(y, af) - auroc(y, ab)) - (auroc(y, bf) - auroc(y, bb))
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            bs.append((auroc(y[idx], af[idx]) - auroc(y[idx], ab[idx]))
                      - (auroc(y[idx], bf[idx]) - auroc(y[idx], bb[idx])))
        except Exception:
            pass
    if not bs:
        return None
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(d0), 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": len(bs)}


def boot_d(y, oa, ob, genes, n_boot=1000, seed=11):
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
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(d0), 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": len(bs)}


def build_codon_freq():
    d = json.load(open(CDS_JSON))
    lut = np.full(256, 255, dtype=np.uint8)
    for i, b in enumerate("TCAG"):
        lut[ord(b)] = i; lut[ord(b.lower())] = i
    cnt = np.zeros(64, dtype=np.int64)
    for k in d:
        a = np.frombuffer(d[k].encode("ascii", "ignore"), dtype=np.uint8)
        a = a[:len(a) - len(a) % 3]
        if len(a) < 3:
            continue
        m = a.reshape(-1, 3)
        m = m[(lut[m] != 255).all(1)]
        if len(m) == 0:
            continue
        cnt += np.bincount(lut[m[:, 0]].astype(np.int64) * 16
                           + lut[m[:, 1]] * 4 + lut[m[:, 2]], minlength=64)
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
    return {"freq": freq, "rank": rank, "n_tx": len(d), "total": int(cnt.sum())}


def cheap_seq(seq_arm, ref_seq, vpos, off, cf):
    """序列类廉价特征（不含位置）：8 连续 + 16 碱基变化 one-hot。"""
    n = len(seq_arm)
    F = np.zeros((n, 24))
    for i in range(n):
        s = str(seq_arm[i]); r = str(ref_seq[i])
        st = vpos[i] - off[i]
        if st < 0 or st + 3 > len(s) or st + 3 > len(r):
            continue
        ca = s[st:st + 3].upper(); cr = r[st:st + 3].upper()
        d = [j for j in range(3) if cr[j] != ca[j]]
        if len(d) != 1:
            continue
        j = d[0]; br, ba = cr[j], ca[j]
        if br not in BASES or ba not in BASES:
            continue
        ia, ib = IDX_OF.get(ca, -1), IDX_OF.get(cr, -1)
        fa = cf["freq"][ia] if ia >= 0 else 0.0
        fr = cf["freq"][ib] if ib >= 0 else 0.0
        ra = cf["rank"][ia] if ia >= 0 else 0.0
        rr = cf["rank"][ib] if ib >= 0 else 0.0
        gc = (1 if ba in "GC" else 0) - (1 if br in "GC" else 0)
        pur = lambda b: 1 if b in "AG" else 0
        oh = [0.0] * 16
        if br != ba:
            oh[BASES.index(br) * 4 + BASES.index(ba)] = 1.0
        F[i] = [fa - fr, abs(fa - fr), float(gc), float(pur(br) == pur(ba)),
                rr - ra, float(len(FAMILY.get(AA_OF.get(cr, "*"), []))), fa, fr] + oh
    return F


def run_scope(tag, m, D_r, D_c, C_r, C_c, pos, y, fold_id, genes, boot):
    print(f"\n--- 口径 {tag}  n={int(m.sum())} ---", flush=True)
    Dr, Dc, Cr, Cc = D_r[m], D_c[m], C_r[m], C_c[m]
    pp = pos[m].reshape(-1, 1)
    yy, ff, gg = y[m], fold_id[m], genes[m]
    nr = np.linalg.norm(Dr, axis=1).reshape(-1, 1)
    nc = np.linalg.norm(Dc, axis=1).reshape(-1, 1)

    O = {}
    O["r0_full"] = oof(Dr, yy, ff)
    O["c0_full"] = oof(Dc, yy, ff)
    O["r1_pos_full"] = oof(np.column_stack([pp, Dr]), yy, ff)
    O["c1_pos_full"] = oof(np.column_stack([pp, Dc]), yy, ff)
    O["r1_pos_base"] = oof(pp, yy, ff)
    O["r2_seq_full"] = oof(np.column_stack([pp, Cr, Dr]), yy, ff)
    O["c2_seq_full"] = oof(np.column_stack([pp, Cc, Dc]), yy, ff)
    O["r2_seq_base"] = oof(np.column_stack([pp, Cr]), yy, ff)
    O["c2_seq_base"] = oof(np.column_stack([pp, Cc]), yy, ff)
    O["r3_all_full"] = oof(np.column_stack([pp, Cr, nr, Dr]), yy, ff)
    O["c3_all_full"] = oof(np.column_stack([pp, Cc, nc, Dc]), yy, ff)
    O["r3_all_base"] = oof(np.column_stack([pp, Cr, nr]), yy, ff)
    O["c3_all_base"] = oof(np.column_stack([pp, Cc, nc]), yy, ff)

    aucs = {k: round(float(auroc(yy, v)), 4) for k, v in O.items()}
    print("  AUC:", json.dumps(aucs), flush=True)

    out = {"n": int(m.sum()), "auc": aucs, "increments": {}}
    levels = [
        ("L0 无基线", None, "r0_full", "c0_full"),
        # 位置不依赖臂，两臂共用同一个基线（同一条变异的位置只有一个）
        ("L1 控制位置", ("r1_pos_base", "r1_pos_base"), "r1_pos_full", "c1_pos_full"),
        ("L2 控制位置+序列特征", ("r2_seq_base", "c2_seq_base"), "r2_seq_full", "c2_seq_full"),
        ("L3 再控制幅度", ("r3_all_base", "c3_all_base"), "r3_all_full", "c3_all_full"),
    ]
    print("  --- 增量（真实臂 / 对照臂 / 二者之差）---", flush=True)
    for lab, base, rf, cf_ in levels:
        if base is None:
            ir = auroc(yy, O["r0_full"]) - 0.5
            ic = auroc(yy, O["c0_full"]) - 0.5
            dd = boot_d(yy, O["r0_full"], O["c0_full"], gg, n_boot=boot)
        else:
            ir = auroc(yy, O[rf]) - auroc(yy, O[base[0]])
            ic = auroc(yy, O[cf_]) - auroc(yy, O[base[1]])
            dd = boot_dd(yy, O[rf], O[base[0]], O[cf_], O[base[1]], gg, n_boot=boot)
        out["increments"][lab] = {
            "incr_real": round(float(ir), 4), "incr_ctrl": round(float(ic), 4),
            "diff": dd["delta"], "ci_lo": dd["ci_lo"], "ci_hi": dd["ci_hi"],
            "excludes_zero": dd["excludes_zero"]}
        print(f"    {lab:22s} 真实 +{ir:.4f} | 对照 +{ic:.4f} | 差 {dd['delta']:+.4f} "
              f"CI [{dd['ci_lo']}, {dd['ci_hi']}] 排除零={dd['excludes_zero']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    print(f"=== 增量分解 | {a.model} ===", flush=True)

    z = np.load(OUT / f"e1_psecenter_emb_{a.model}.npz", allow_pickle=True)
    D_r = z["alt"] - z["ref"]; D_c = z["psecenter"] - z["ref"]
    y = np.asarray(z["labels"], int); fold_id = np.asarray(z["fold_id"], int)
    gi = np.asarray(z["global_index"], int)
    fx0 = pd.read_csv(FRZ / "dataset_v2" / "variants.tsv", sep="\t")
    genes = np.asarray(np.load(OUT / "e1s_genes.npy", allow_pickle=True))[gi]

    S_ref = np.load(OUT / "e1s_ref.npy", allow_pickle=True)
    S_alt = np.load(OUT / "e1s_alt.npy", allow_pickle=True)
    S_pse = np.load(OUT / "e1s_psecenter.npy", allow_pickle=True)
    cpos_all = pd.to_numeric(fx0["cds_position"], errors="coerce").values
    off = np.where(np.isnan(cpos_all), -1, ((cpos_all - 1) % 3).astype(int))[gi]
    vpos = np.full(len(gi), -1, dtype=int)
    for i, g in enumerate(gi):
        d = [j for j, (x, yy) in enumerate(zip(str(S_ref[g]), str(S_alt[g]))) if x != yy]
        if len(d) == 1:
            vpos[i] = d[0]
    if (vpos < 0).any() or (off < 0).any():
        die("相位或变异位未定")

    cf = build_codon_freq()
    C_r = cheap_seq([str(S_alt[g]) for g in gi], [str(S_ref[g]) for g in gi], vpos, off, cf)
    C_c = cheap_seq([str(S_pse[g]) for g in gi], [str(S_ref[g]) for g in gi], vpos, off, cf)
    pos = pd.to_numeric(fx0.iloc[gi]["exon_boundary_distance"],
                        errors="coerce").fillna(0).values
    zero = np.linalg.norm(D_c, axis=1) < 1e-12

    res = {"model": a.model, "n_total": int(len(y)),
           "n_control_zero": int(zero.sum())}
    res["scope_all"] = run_scope("A 全样本", np.ones(len(y), bool),
                                 D_r, D_c, C_r, C_c, pos, y, fold_id, genes, a.boot)
    res["scope_control_constructed"] = run_scope(
        "B 对照已构造", ~zero, D_r, D_c, C_r, C_c, pos, y, fold_id, genes, a.boot)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    op = OUT / f"e1_decompose_incr_{a.model}.json"
    json.dump(res, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n写入 {op}  用时 {res['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
