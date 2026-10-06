# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
EXP-010｜标签地基检验：按 ClinVar review status 分层重跑响应向量主分析

为什么做
--------
§十一.R4 判定为**最大方向性风险**：若 ClinVar 同义致病标签本身有结构性噪声，
主结果 +0.085 可能是噪声结构而非生物学信号，整套实验白跑。

不占 GPU：嵌入已落盘，只重跑探针与 bootstrap。
评估函数**逐字复用** WB-e1-multimodel-fix.py（oof_scores / auc_ci / paired_delta），
保证与 EXP-007b 口径一致（5 折、StandardScaler+LR(C=1.0)、按基因聚类 bootstrap、
seed 7 / 11）。

分层（先写死，避免事后挑层）
----------------------------
L_all   : 全部 usable 子集
L_noNA  : 剔除 "no assertion criteria provided"（最低质量档）
L_high  : 仅 "multiple submitters, no conflicts" + "reviewed by expert panel"

判读标准（先写死）
------------------
主结论（某模型的 Δ_alt 显著非零）必须在 **L_noNA 与 L_high 上同向**，否则：
  - 只在 L_all 显著、分层后消失  ⇒ 信号来自低质量标签，主结果作废；
  - 三层同向且 CI 均不含 0      ⇒ 标签地基通过；
  - L_high 样本不足导致 CI 过宽 ⇒ 只能作描述性，不得进主结果。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

# ---- 与 EXP-007b 逐字一致 ----
def oof_scores(X, y, groups, split):
    X = np.asarray(X, dtype=np.float64)
    oof = np.zeros(len(y))
    folds = (list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)) if split == "random"
             else list(GroupKFold(5).split(X, y, groups)))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def auc_ci(y, s, groups, n_boot=1000, seed=7):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y[idx], s[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def paired_delta(y, groups, Xa, Xb, split, n_boot=1000, seed=11):
    oa = oof_scores(Xa, y, groups, split)
    ob = oof_scores(Xb, y, groups, split)
    da, db = roc_auc_score(y, oa), roc_auc_score(y, ob)
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y[idx], oa[idx]) - roc_auc_score(y[idx], ob[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc_A": round(float(da), 4), "auc_B": round(float(db), 4),
            "delta": round(float(da - db), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}
# --------------------------------

usable = np.load(OUT / "e1s_usable.npy").astype(bool)
lab_all = np.load(OUT / "e1s_labels.npy")
gen_all = np.load(OUT / "e1s_genes.npy", allow_pickle=True).astype(str)
rs_all = np.load(OUT / "e1s_review_status.npy", allow_pickle=True).astype(str)

# usable 内的索引
idx_all = np.where(usable)[0]
lab, gen, rs = lab_all[idx_all], gen_all[idx_all], rs_all[idx_all]

HIGH = {"criteria provided, multiple submitters, no conflicts",
        "reviewed by expert panel"}
NA = "no assertion criteria provided"

LAYERS = {
    "L_all":  np.ones(len(idx_all), dtype=bool),
    "L_noNA": rs != NA,
    "L_high": np.array([r in HIGH for r in rs]),
}

MODELS = [
    ("codonbert", "clm"),
    ("esm1b-650m", "plm"),
    ("esm2-8m", "plm"),
]

emb = {}
for mid, kind in MODELS:
    try:
        emb[mid] = {
            "ref": np.load(OUT / ("e1s_%s_ref.npy" % mid)),
            "alt": np.load(OUT / ("e1s_%s_alt.npy" % mid)),
            "pse": np.load(OUT / ("e1s_%s_pse.npy" % mid)),
        }
        print("[load] %-12s ref%s alt%s pse%s" % (
            mid, emb[mid]["ref"].shape, emb[mid]["alt"].shape, emb[mid]["pse"].shape))
    except Exception as e:
        print("[skip] %-12s %s" % (mid, str(e)[:80]))

res = {"_layers": {}, "_models": {}}
print("\n" + "=" * 84)
print("%-8s %-12s %6s %6s %5s | %-9s %9s %9s | %s" % (
    "层", "模型", "n", "基因", "正例", "划分", "AUC(diff)", "Δ_alt", "CI 是否含 0"))
print("=" * 84)

for lname, mask in LAYERS.items():
    ii = idx_all[mask]
    y = lab_all[ii].astype(int)
    g = gen_all[ii]
    res["_layers"][lname] = {
        "n": int(len(ii)), "genes": int(len(set(g.tolist()))),
        "pos": int(y.sum()), "neg": int((1 - y).sum())}
    print("%-8s %-12s %6d %6d %5d |" % (lname, "—", len(ii), len(set(g.tolist())), y.sum()),
          "(规模与构成)")

    for mid, kind in MODELS:
        if mid not in emb:
            continue
        E = emb[mid]
        X_ref = E["ref"][mask]
        X_alt = E["alt"][mask]
        X_pse = E["pse"][mask]
        Xd = {"ref": X_ref,
              "diff": X_alt - X_ref,
              "pseudo_diff": X_pse - X_ref}

        row = {}
        for split in ["LOGO", "random"]:
            if split == "LOGO" and len(set(g.tolist())) < 5:
                continue
            a, lo, hi = auc_ci(y, oof_scores(Xd["diff"], y, g, split), g)
            pd_ = paired_delta(y, g, Xd["diff"], Xd["pseudo_diff"], split)
            row[split] = {"diff": {"auc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)},
                          "paired": pd_}
            print("%-8s %-12s %6d %6d %5d | %-9s %9.4f %+9.4f | [%+.4f, %+.4f] %s" % (
                lname, mid, len(ii), len(set(g.tolist())), y.sum(), split, a,
                pd_["delta"], pd_["ci_lo"], pd_["ci_hi"],
                "不含0" if pd_["excludes_zero"] else "**含0**"))
        res["_models"].setdefault(mid, {})[lname] = row
    print("-" * 84)

p = OUT / "exp010_reviewstatus_results.json"
json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[已落盘] %s" % p)

print("\n判读（对照脚本开头写死的标准）：")
for mid, kind in MODELS:
    if mid not in res["_models"]:
        continue
    ds = {l: res["_models"][mid][l].get("LOGO", {}).get("paired", {}).get("delta")
          for l in LAYERS}
    ez = {l: res["_models"][mid][l].get("LOGO", {}).get("paired", {}).get("excludes_zero")
          for l in LAYERS}
    print("  %-12s Δ_alt: %s | 显著: %s" % (
        mid, {k: v for k, v in ds.items()}, {k: v for k, v in ez.items()}))
