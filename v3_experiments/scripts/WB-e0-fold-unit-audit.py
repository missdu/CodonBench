# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-fold-unit-audit.py  (2026-10-03)
目的：审计 LOGO 折的"留出单位"到底是转录本还是基因，并量化同基因泄漏。

背景：
 - V1 主文 L215 声称 "0 of 240 folds share any gene"，L223 称 "holds out all variants of each gene"
 - CD 主文 L117 称 "holds out one transcript at a time"
 - 脚本 run_logo_cv_mlp_v2.py L158: gene_ids.append(row["tx_id"])  <-- 变量名 gene，装的是 tx

本脚本回答三个问题：
 Q1 折定义单位是什么（读 npz 键名）
 Q2 有多少基因拥有 >=2 个转录本
 Q3 留一转录本时，有多少 valid fold 的测试集与训练集共享基因（即泄漏）
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import json, numpy as np
from collections import defaultdict

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp") + "/results/unified_eval/synpath_data"

z = np.load(f"{BASE}/split_logo_cv.npz", allow_pickle=True)

# ---- Q1 折定义单位 ----
tx_keys = [k for k in z.files if k.startswith("tx_")]
print("Q1 折定义单位")
print("   unique_tx_ids 长度 :", z["unique_tx_ids"].shape[0])
print("   tx_* 索引键数量    :", len(tx_keys))
print("   n_valid_folds      :", int(z["n_valid_folds"]))
vf = z["valid_fold_tx_ids"]
print("   valid_fold_tx_ids  :", vf.shape, vf.dtype)
print("   样例               :", list(vf[:3]) if vf.ndim else vf)
print()

# ---- Q2 基因 -> 转录本 ----
# ⚠️ 坑（2026-10-03 首次跑踩到）：json 顶层是"元数据 + 映射"的混合结构，
#    直接遍历顶层会把 n_variants(int) 等元数据当映射读，得到 tx2gene 为空的
#    **静默假阴性**。真正的映射在 m["tx_to_gene"] / m["gene_to_tx_list"]。
m = json.load(open(f"{BASE}/gene_transcript_map.json"))
print("Q2 基因-转录本映射")
print("   json 顶层键:", list(m.keys()))
for k in ("n_variants", "n_unique_tx_ids", "n_unique_gene_symbols",
          "n_multi_transcript_genes", "n_variants_in_multi_tx_genes",
          "pct_variants_in_multi_tx_genes"):
    if k in m:
        print(f"   {k:32s}: {m[k]}")

assert "tx_to_gene" in m, "映射键缺失，脚本需修"
tx2gene = dict(m["tx_to_gene"])
gene2tx = {g: set(txs) for g, txs in m.get("gene_to_tx_list", {}).items()}
if not tx2gene:
    raise SystemExit("FATAL: tx2gene 为空 —— 静默假阴性，拒绝出结论")

print("   转录本数(有基因):", len(tx2gene))
print("   基因数          :", len(gene2tx))
multi = {g: txs for g, txs in gene2tx.items() if len(txs) > 1}
print("   拥有 >=2 转录本的基因数:", len(multi))
if multi:
    show = list(multi.items())[:5]
    print("   样例:", show)
print()

# ---- Q3 逐折泄漏 ----
print("Q3 留一转录本下的同基因泄漏")
valid_tx = [str(t) for t in (vf.tolist() if vf.ndim else [vf])]
all_tx = [str(t) for t in z["unique_tx_ids"]]

n_leak_folds = 0
leak_detail = []
for t in valid_tx:
    g = tx2gene.get(t)
    if g is None:
        continue
    train_tx = set(all_tx) - {t}
    # 训练集里同一基因的其他转录本
    same_gene_train = [x for x in gene2tx.get(g, set()) if x in train_tx and x != t]
    if same_gene_train:
        n_leak_folds += 1
        leak_detail.append((t, g, same_gene_train))

print(f"   valid folds 总数           : {len(valid_tx)}")
print(f"   存在同基因泄漏的折数       : {n_leak_folds}")
print(f"   无泄漏的折数               : {len(valid_tx) - n_leak_folds}")
if leak_detail:
    print("   泄漏折样例（前 5）:")
    for t, g, sgt in leak_detail[:5]:
        print(f"      held-out tx={t}  gene={g}  训练集同基因转录本={sgt}")
print()

# ---- 结论 ----
print("结论")
if n_leak_folds == 0:
    print("   留一转录本 == 留一基因：无同基因泄漏，V1 '0 gene overlap' 成立。")
else:
    print(f"   留一转录本 != 留一基因：{n_leak_folds} 个折存在同基因泄漏，")
    print("   V1 主文 '0 of 240 folds share any gene' 不成立，须改口径或改措辞。")
