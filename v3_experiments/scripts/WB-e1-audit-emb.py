# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— 现有嵌入缓存退化审计

要回答的三个问题（决定 E1/E2/E3/E5 全部结果能不能用）：
  Q1 现有缓存 codonbert_synonymous_emb.npy 是否退化？
     判据：逐维标准差、任意两行余弦相似度、唯一行数
  Q2 rand_variant_only / rand_all 两个对照是否与原嵌入可区分？
     若三者基本相同 ⇒ 随机化对照是空转，E1 的"变异特异增量"无意义
  Q3 与新提的 e1v2_*（T→U 修正后）相比，差异有多大？
     判据：同下标行的余弦相似度（对齐后）

只读，不改任何文件。
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

import numpy as np
from pathlib import Path

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
OUT = SUPP / "wb_rerun"


def stat(name, E):
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(-1, 1)
    print("\n== %s ==" % name)
    print("  shape %s | dtype %s" % (E.shape, E.dtype))
    sd = E.std(axis=0)
    print("  逐维标准差: mean %.6g | min %.6g | max %.6g"
          % (sd.mean(), sd.min(), sd.max()))
    nzero = int((sd < 1e-12).sum())
    print("  常数维度(标准差<1e-12): %d / %d" % (nzero, E.shape[1]))
    # 唯一行
    try:
        u = np.unique(np.round(E, 8), axis=0)
        print("  唯一行数: %d / %d" % (len(u), E.shape[0]))
    except Exception as e:
        print("  唯一行统计失败:", e)
    # 行间余弦相似度（抽样 200×200）
    m = min(200, E.shape[0])
    A = E[:m]
    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    nrm[nrm == 0] = 1
    An = A / nrm
    C = An @ An.T
    off = C[~np.eye(m, dtype=bool)]
    print("  行间余弦: mean %.6f | min %.6f | max %.6f"
          % (off.mean(), off.min(), off.max()))
    return E


def cos_pairs(A, B):
    """逐行余弦相似度"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)
    na[na == 0] = 1
    nb[nb == 0] = 1
    return (A * B).sum(1) / (na * nb)


files = {
    "cache_C_alt": SUPP / "codonbert_synonymous_emb.npy",
    "cache_C_rand_variant": SUPP / "codonbert_synonymous_rand_variant_only_emb.npy",
    "cache_C_rand_all": SUPP / "codonbert_synonymous_rand_all_emb.npy",
}

loaded = {}
for k, p in files.items():
    if p.exists():
        loaded[k] = stat(k, np.load(p, allow_pickle=True))
    else:
        print("\n== %s == 文件不存在: %s" % (k, p))

print("\n\n########## 对照可分性 ##########")
if "cache_C_alt" in loaded and "cache_C_rand_variant" in loaded:
    c = cos_pairs(loaded["cache_C_alt"], loaded["cache_C_rand_variant"])
    print("C_alt vs C_rand_variant 逐行余弦: mean %.6f | min %.6f"
          % (c.mean(), c.min()))
    d = np.abs(loaded["cache_C_alt"] - loaded["cache_C_rand_variant"]).mean()
    print("  mean|差| = %.6g" % d)
if "cache_C_alt" in loaded and "cache_C_rand_all" in loaded:
    c = cos_pairs(loaded["cache_C_alt"], loaded["cache_C_rand_all"])
    print("C_alt vs C_rand_all    逐行余弦: mean %.6f | min %.6f"
          % (c.mean(), c.min()))
    d = np.abs(loaded["cache_C_alt"] - loaded["cache_C_rand_all"]).mean()
    print("  mean|差| = %.6g" % d)

print("\n########## 新提嵌入（T→U 修正后）##########")
new_ref_p = OUT / "e1v2_codonbert_ref.npy"
new_alt_p = OUT / "e1v2_codonbert_alt.npy"
if new_ref_p.exists():
    Nr = stat("e1v2_ref", np.load(new_ref_p))
    Na = stat("e1v2_alt", np.load(new_alt_p))
    c = cos_pairs(Nr, Na)
    print("\n新提 ref vs alt 逐行余弦: mean %.6f | min %.6f" % (c.mean(), c.min()))
    print("新提 mean|ref-alt| = %.6g" % np.abs(Nr - Na).mean())
    if "cache_C_alt" in loaded:
        Ca = loaded["cache_C_alt"]
        if Ca.shape == Nr.shape:
            c2 = cos_pairs(Ca, Nr)
            print("缓存 C_alt vs 新提 ref 逐行余弦: mean %.6f | min %.6f"
                  % (c2.mean(), c2.min()))
        else:
            print("形状不一致，跳过直接比对: cache %s vs new %s"
                  % (Ca.shape, Nr.shape))
else:
    print("新提嵌入不存在:", new_ref_p)

print("\n审计完成。")
