# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e4a-fixup.py (2026-10-03)
补算三件在 WB-e4a-gene-perm.py 里做错或没做的事，不重跑模型（省算力）：

1. 🔴 G1 的 SEM 被低估 —— 脚本把 5 次重复 × 5 折 = 25 个折当 25 个独立样本算 SEM。
   实际上只有 5 个独立划分，同一样本被重复测试 ⇒ SEM 虚假地小（0.0033）。
   Reject §五.3 点名的正是这类"低估基因内相关造成的不确定性"。
   正确做法：先按 rep 聚合（每 rep 一个均值），再在 5 个 rep 上算 SEM。

2. 🔴 置换覆盖率的分子分母写反 —— 脚本 append 的是 (multi, ngene)，
   却用 mul=sum(ngene)、tot=sum(multi) ⇒ 打印的 0.176 是"基因数/多样本样本数"，无意义。
   正确：coverage = 训练集中"所属基因在训练集内有 ≥2 条变异"的样本占比。

3. G2/G1 下 real − perm 的**配对差值及其 CI**（逐折相减），
   这是 E4-A 真正该报的量：置换到底拿走了多少性能。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import json, numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
OUT = f"{BASE}/results/supplementary/wb_rerun/wb_frozen"
r = json.load(open(f"{OUT}/e4a_geneperm_v1.json"))

df = pd.read_parquet(f"{BASE}/results/unified_eval/synpath_data/synpath_variants.parquet")
y = df["label"].values.astype(int); genes = df["gene_symbol"].values; n = len(df)
d = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fold_id = d["fold_id"]; gidx = d["global_index"]; N240 = int(fold_id.max()) + 1

print("=" * 72)
print("1. 置换覆盖率（修正分子分母）")
print("=" * 72)
covs = []
for k in range(N240):
    te = gidx[fold_id == k]
    tr = np.setdiff1d(np.arange(n), te)
    g_tr = genes[tr]
    sz = pd.Series(g_tr).value_counts()
    multi_genes = set(sz[sz > 1].index)
    cov = np.mean([g in multi_genes for g in g_tr])
    covs.append(cov)
covs = np.array(covs)
print(f"  未见基因场景（240 folds）训练集：")
print(f"    每条折：所属基因在训练集内有 >=2 条变异的样本占比")
print(f"    min={covs.min():.4f} 中位={np.median(covs):.4f} 均值={covs.mean():.4f} max={covs.max():.4f}")
print(f"  ⇒ 此前打印的 0.176 是错的（分子分母颠倒），实际为 {covs.mean():.4f}")

print()
print("=" * 72)
print("2. real − perm 配对差值（逐折相减）")
print("=" * 72)
for scen in ("G2_unseen_gene", "G1_known_gene"):
    if scen not in r["scenarios"]:
        continue
    print(f"\n  [{scen}]")
    S = r["scenarios"][scen]
    for f in ("B2_background", "B3_codon_table"):
        kr, kp = f"{f}|real", f"{f}|perm"
        if kr not in S or kp not in S:
            continue
        a = np.array(S[kr]["auc_per_fold"], dtype=float)
        b = np.array(S[kp]["auc_per_fold"], dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        dd = a[m] - b[m]
        se = dd.std(ddof=1) / np.sqrt(m.sum())
        print(f"    {f:16s} real={a[m].mean():.4f} perm={b[m].mean():.4f} "
              f"差={dd.mean():+.4f} 95%CI[{dd.mean()-1.96*se:+.4f},{dd.mean()+1.96*se:+.4f}] "
              f"(n={m.sum()} 折)")
        S[f"{f}|paired_drop"] = {"mean": float(dd.mean()),
                                 "ci95": [float(dd.mean() - 1.96 * se), float(dd.mean() + 1.96 * se)],
                                 "n_folds": int(m.sum())}

print()
print("=" * 72)
print("3. G1 的 SEM 修正（按 rep 聚合，5 个独立划分）")
print("=" * 72)
NREP, NFOLD = 5, 5
if "G1_known_gene" in r["scenarios"]:
    S = r["scenarios"]["G1_known_gene"]
    for f in ("B2_background", "B3_codon_table"):
        for tag in ("real", "perm"):
            k = f"{f}|{tag}"
            if k not in S:
                continue
            arr = np.array(S[k]["auc_per_fold"], dtype=float)
            if len(arr) != NREP * NFOLD:
                print(f"    {k}: 长度 {len(arr)} != 25，跳过"); continue
            per_rep = arr.reshape(NREP, NFOLD).mean(axis=1)
            sem_rep = per_rep.std(ddof=1) / np.sqrt(NREP)
            print(f"    {k:26s} 折级 SEM={S[k]['auc_sem']:.4f}  →  rep 级 SEM={sem_rep:.4f}"
                  f"   均值 {per_rep.mean():.4f}")
            S[k]["auc_sem_by_rep"] = float(sem_rep)
            S[k]["auc_mean_by_rep"] = float(per_rep.mean())
            S[k]["per_rep"] = per_rep.tolist()

json.dump(r, open(f"{OUT}/e4a_geneperm_v1.json", "w"), indent=2, ensure_ascii=False)
print("\n已回写", f"{OUT}/e4a_geneperm_v1.json")
