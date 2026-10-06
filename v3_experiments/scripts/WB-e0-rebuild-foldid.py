# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-rebuild-foldid.py  (2026-10-03)
目的：判断 CD 线现成的 LOGO per-fold 预测能否直接用于 WB 的逐折配对分析，
      即能否从 split_logo_cv.npz 重建 fold_id，从而**免掉一轮 GPU 重跑**。

背景：
 - CD 线产物 logo_cv_predictions_<model>_task3_synonymous.npz 只有
   (lr_y_true, lr_y_proba, mlp_y_true, mlp_y_proba)，各 1847，**没有 fold_id**
 - 折定义在 results/unified_eval/synpath_data/split_logo_cv.npz：
   unique_tx_ids(497) + tx_<ID>_indices(497) + valid_fold_tx_ids(240)

方法：每个 valid fold = 一个 held-out 转录本，该转录本的所有变异即该折的测试集。
     fold_id[i] = 样本 i 所属 valid tx 在 valid_fold_tx_ids 中的序号。

判据（三条全过才算"可用"）：
 P1 重建出的有效样本总数 == 1847（与 CD 预测 npz 的长度一致）
 P2 每个折内正/负样本都 >= 1（与"fold valid iff 每类至少一个"的定义一致）
 P3 每折样本数之和 == 1847，且折数 == 240
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import numpy as np

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp") + "/results/unified_eval/synpath_data"
SUP = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp") + "/results/supplementary"

z = np.load(f"{BASE}/split_logo_cv.npz", allow_pickle=True)
valid_tx = [str(t) for t in z["valid_fold_tx_ids"].tolist()]
print(f"valid folds (tx): {len(valid_tx)}")

# 收集每个 valid fold 的样本索引
fold_indices = []
for t in valid_tx:
    key = f"tx_{t}_indices"
    if key not in z.files:
        print(f"   !! 缺键 {key}")
        fold_indices.append(np.array([], dtype=int))
    else:
        fold_indices.append(np.asarray(z[key], dtype=int))

n_per_fold = np.array([len(x) for x in fold_indices])
total = int(n_per_fold.sum())
print(f"P1 重建样本总数 : {total}   (CD 预测 npz 长度 = 1847)")
print(f"P3 折数         : {len(fold_indices)}")
print(f"   每折样本数   : min={n_per_fold.min()} median={int(np.median(n_per_fold))} max={n_per_fold.max()}")

ok1 = (total == 1847)
ok3 = (len(fold_indices) == 240) and (total == int(n_per_fold.sum()))
print(f"   P1 {'PASS' if ok1 else 'FAIL'} / P3 {'PASS' if ok3 else 'FAIL'}")
print()

# 用 CD 的真实预测验证 P2：每折正/负都 >=1
pred = np.load(f"{SUP}/logo_cv_predictions_codonbert_task3_synonymous.npz", allow_pickle=True)
y = pred["mlp_y_true"]
print(f"CD 预测 npz: y_true 长度 = {len(y)}, 正样本 = {int(y.sum())}, 负样本 = {int((1-y).sum())}")

# 🔴 2026-10-03 发现：tx_*_indices 是在**全集 2840** 上的索引，不是 1847 上的。
#    用 total 建数组会越界。全集大小从 gene_transcript_map.json 的 n_variants 取。
import json
_nv = json.load(open(f"{BASE}/gene_transcript_map.json"))["n_variants"]
print(f"全集 n_variants = {_nv}")

# 🔴 类别平衡检查（Reject §二.3.2：排除折会改变评价人群）
print(f"1847 子集类别: 正={int(y.sum())} 负={int((1-y).sum())} "
      f"正样本率={y.mean():.3f}")
print("   对照：全集 2840 为 1420/1420 平衡 ⇒ 子集已不平衡，须在论文声明")

# pooled AUC 自检：能否复现 CD 表 2 的 CodonBERT MLP = 0.564
from sklearn.metrics import roc_auc_score
pooled = roc_auc_score(y, pred["mlp_y_proba"])
print(f"pooled AUC(CodonBERT MLP) = {pooled:.4f}   CD 表 2 记 0.564")

if ok1:
    fold_id = np.full(_nv, -1, dtype=int)
    for k, idx in enumerate(fold_indices):
        fold_id[idx] = k
    # 注意：fold_indices 的拼接顺序未必等于 npz 中预测的顺序，
    # 这里只验证"数量结构"能否支持逐折配对，不声称逐位对齐。
    # 🔑 假设 H：npz 中 1847 个样本 = 按 valid fold 顺序依次拼接
    #    依据 run_logo_cv_pooled_auc.py L111-112: all_y_true.extend(y_te.tolist())
    #    逐折遍历、跳过无效折 ⇒ fold_id = repeat(折号, 该折 n_test)
    fid = np.repeat(np.arange(len(n_per_fold)), n_per_fold)
    print(f"H  fold_id 长度 = {len(fid)}  (应为 1847)  -> {'OK' if len(fid)==len(y) else 'MISMATCH'}")
    pos_per_fold = np.array([int(y[fid == k].sum()) for k in range(len(n_per_fold))])
    neg_per_fold = np.array([int((1 - y[fid == k]).sum()) for k in range(len(n_per_fold))])
    bad = int(((pos_per_fold < 1) | (neg_per_fold < 1)).sum())
    print(f"P2 折内正/负均 >=1 : {len(n_per_fold) - bad}/{len(n_per_fold)} 通过, {bad} 折不满足")
    ok2 = (bad == 0) and (len(fid) == len(y))
    print(f"   P2 {'PASS' if ok2 else 'FAIL'}")
    if ok2:
        print(f"   逐折 AUC（前 5 折）:")
        for k in range(5):
            mk = fid == k
            a = roc_auc_score(y[mk], pred["mlp_y_proba"][mk]) if len(set(y[mk])) > 1 else float('nan')
            print(f"      fold {k}: n={mk.sum()} pos={pos_per_fold[k]} neg={neg_per_fold[k]} AUC={a:.4f}")
        pf = np.array([roc_auc_score(y[fid==k], pred["mlp_y_proba"][fid==k])
                       if len(set(y[fid==k]))>1 else np.nan for k in range(len(n_per_fold))])
        print(f"   逐折均值 AUC = {np.nanmean(pf):.4f}  (pooled = {pooled:.4f})")
else:
    ok2 = False
    print("P2 跳过（P1 未过，索引无法对应）")

print()
print("=== 判定 ===")
if ok1 and ok2 and ok3:
    print("可用：fold_id 可从 split_logo_cv.npz 重建，CD 现成预测支持逐折配对分析。")
    print("      ⇒ WB 无需为『拿到 240 折逐折预测』重跑 GPU。")
else:
    print("不可用或不完全可用：见上 FAIL 项，需先修索引对齐再谈复用。")
    print("      ⚠️ 注意：本脚本只验证『数量结构』，未验证『样本顺序逐位对齐』；")
    print("         真正复用前必须再做一次顺序对齐校验（用模型 AUC 复算比对）。")
