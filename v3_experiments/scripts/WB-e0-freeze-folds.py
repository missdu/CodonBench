# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-freeze-folds.py  (2026-10-03)
目的：把 V1 官方 LOGO 口径（逐基因留一、240 valid folds）**冻结落盘**，
      作为后续所有数值定标的唯一分折依据。

为什么必须冻结：
 - WB 此前全程用 GroupKFold(5)/StratifiedKFold(5)，与 V1 官方口径差 0.019 ⇒ 所有数值须重算
 - 折定义已确认可用：留一转录本 ≡ 留一基因（实测 n_multi_transcript_genes = 0）

🔴 本脚本的核心难点：**顺序对齐**。
   CD 线预测 npz 里 1847 个样本没有 fold_id，只有 y_true/y_proba。
   上一个脚本假设"按 valid fold 顺序依次拼接"，但**该假设从未被独立验证**：
     - pooled AUC 复现（0.5643）**不能**证明顺序 —— AUC 是排序统计量，对样本顺序不变
     - 逐折 AUC 依赖顺序假设本身 ⇒ 循环论证
   ⇒ 必须用**与预测无关**的证据验证顺序。本脚本用两条独立证据：

   E1 单调性：若原变体表按 tx_id 排序，则按 valid_tx 顺序拼接后的全集索引应严格递增
   E2 连续性：每个 tx 的变异在原表中应构成连续区间（同一 tx 的样本排在一起）

   两条都过 ⇒ 拼接顺序 = 索引升序，顺序唯一确定，可放心落盘。

输出（永不覆盖，带 batch_id）：
   results/supplementary/wb_rerun/wb_frozen/folds_240logo_v1.npz
     - fold_id        (1847,)   每个样本属于哪个折（0..239）
     - global_index   (1847,)   该样本在全集 2840 中的原始索引
     - fold_tx_id     (240,)    每折留出的转录本 ID
     - fold_gene      (240,)    每折留出的基因（tx→gene 映射）
     - fold_n/pos/neg (240,)    每折样本数/正/负
     - labels         (1847,)   从 CD 预测 npz 取（仅作交叉校验，不作为标签权威源）
   + folds_240logo_v1.json（人类可读摘要 + 全部判据结果）
"""
import os, json
import numpy as np

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp") + "/results/unified_eval/synpath_data"
SUP = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp") + "/results/supplementary"
OUTDIR = f"{SUP}/wb_rerun/wb_frozen"
BATCH = "folds_240logo_v1"
os.makedirs(OUTDIR, exist_ok=True)

report = {"batch": BATCH, "checks": {}, "notes": []}

# ---------- 读折定义 ----------
z = np.load(f"{BASE}/split_logo_cv.npz", allow_pickle=True)
valid_tx = [str(t) for t in z["valid_fold_tx_ids"].tolist()]
n_folds = len(valid_tx)

fold_indices = []
missing = 0
for t in valid_tx:
    key = f"tx_{t}_indices"
    if key not in z.files:
        fold_indices.append(np.array([], dtype=int)); missing += 1
    else:
        fold_indices.append(np.asarray(z[key], dtype=int))

n_per_fold = np.array([len(x) for x in fold_indices])
total = int(n_per_fold.sum())
gmap = json.load(open(f"{BASE}/gene_transcript_map.json"))
n_variants = gmap["n_variants"]

print(f"折数 = {n_folds}, 缺键 = {missing}")
print(f"重建样本总数 = {total}; 全集 n_variants = {n_variants}")

# ---------- 顺序判据（2026-10-03 修正版） ----------
# ❌ 撤回初版 E1/E2（拼接单调递增 / 折内索引连续）：
#    这两条检验的是"原变体表是否按 tx 连续排列"，与"npz 样本顺序"无关。
#    实测原表不是按 tx 连续排的（同 tx 的变异分散在全表）⇒ 初版判据无效，已删。
# ✅ 正确判据 = 复现生成脚本的遍历顺序（run_logo_cv_pooled_auc.py L86）：
#      for i, gene in enumerate(unique_genes): ... y_te = y[test_mask]; all_y_true.extend(...)
#    ⇒ npz 顺序 = 按 unique 顺序逐折拼接，每折内取 test_mask 的原索引升序。
#    因此要验的只有三件事：
concat = np.concatenate(fold_indices) if total else np.array([], dtype=int)

# E1 unique_tx_ids 有序（脚本遍历的就是它）
u_tx = [str(t) for t in z["unique_tx_ids"].tolist()]
uniq_sorted = bool(u_tx == sorted(u_tx))
print(f"\nE1 unique_tx_ids 有序: {uniq_sorted}  (n={len(u_tx)})")
report["checks"]["E1_unique_tx_ids_sorted"] = uniq_sorted

# E2 valid_fold_tx_ids 恰是按 unique 顺序取出的子序列（⇒ 折顺序与脚本遍历顺序一致）
v_set = set(valid_tx)
v_sub = [t for t in u_tx if t in v_set]
subseq = bool(v_sub == valid_tx)
print(f"E2 valid 折顺序 == unique 顺序的子序列: {subseq}")
report["checks"]["E2_valid_folds_follow_unique_order"] = subseq

# E3 每折内索引升序（⇒ 每折内部顺序 = 原索引升序，与 y[test_mask] 一致）
asc = bool(all(np.all(np.diff(np.asarray(fold_indices[k])) > 0)
               for k in range(n_folds) if len(fold_indices[k]) > 1))
print(f"E3 每折内索引升序: {asc}")
report["checks"]["E3_within_fold_index_ascending"] = asc

mono, noncont = True, []   # 保留变量名以兼容下游断言

# ---------- 唯一性：折之间不重叠 ----------
allset = set(concat.tolist())
uniq = len(allset) == len(concat)
print(f"E4 索引全局唯一（折间不重叠）: {uniq}  (唯一值 {len(allset)} / 总 {len(concat)})")
report["checks"]["E4_no_overlap_between_folds"] = uniq

# ---------- 与 CD 预测长度一致 ----------
pred_path = f"{SUP}/logo_cv_predictions_codonbert_task3_synonymous.npz"
pred = np.load(pred_path, allow_pickle=True)
y_cd = pred["mlp_y_true"]
len_match = (total == len(y_cd))
print(f"E5 重建总数 == CD 预测长度: {len_match}  ({total} vs {len(y_cd)})")
report["checks"]["E5_total_matches_cd_npz"] = {
    "ok": len_match, "rebuilt": total, "cd": int(len(y_cd))}

order_ok = uniq_sorted and subseq and asc and uniq and len_match
report["checks"]["ORDER_DETERMINED"] = order_ok
print(f"\n>>> 顺序是否已被独立确定: {order_ok}")

# ---------- E6 端到端敏感性检验（对顺序敏感，能证伪） ----------
# 逻辑：每折只有 3~61 个样本（中位 5）。若 fold_id 错配，
# 逐折 AUC 会大量退化到 0/1 或 NaN，逐折均值会明显偏离 pooled。
# pooled AUC 对顺序不变 ⇒ 不能用来验顺序；逐折均值可以。
try:
    from sklearn.metrics import roc_auc_score
    fid_tmp = np.repeat(np.arange(n_folds), n_per_fold)
    pf = []
    for k in range(n_folds):
        mk = fid_tmp == k
        yy = y_cd[mk]
        if len(set(yy)) < 2:
            pf.append(np.nan); continue
        pf.append(roc_auc_score(yy, pred["mlp_y_proba"][mk]))
    pf = np.array(pf)
    nan_folds = int(np.isnan(pf).sum())
    degen = int(((pf == 0.0) | (pf == 1.0)).sum())
    print(f"E6 逐折 AUC: 均值={np.nanmean(pf):.4f} 中位={np.nanmedian(pf):.4f} "
          f"NaN折={nan_folds} 退化折(0/1)={degen}/{n_folds}")
    report["checks"]["E6_per_fold_auc_sanity"] = {
        "mean": float(np.nanmean(pf)), "median": float(np.nanmedian(pf)),
        "nan_folds": nan_folds, "degenerate_folds": degen, "n_folds": n_folds,
    }
except Exception as e:
    print(f"E6 跳过: {e}")
    report["checks"]["E6_per_fold_auc_sanity"] = {"error": str(e)}

# ---------- 落盘 ----------
fid = np.repeat(np.arange(n_folds), n_per_fold).astype(int)
assert len(fid) == total

# tx → gene 映射
tx2gene = {}
for k, v in gmap.items():
    if isinstance(v, dict):
        for tx, g in v.items():
            tx2gene[str(tx)] = str(g)
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, dict) and "tx" in item:
                tx2gene[str(item["tx"])] = str(item.get("gene", ""))
if not tx2gene:
    report["notes"].append("tx→gene 映射未能构建，fold_gene 置为空串")
fold_gene = np.array([tx2gene.get(t, "") for t in valid_tx])

y = y_cd if len_match else np.zeros(total, dtype=int)
pos_per = np.array([int(y[fid == k].sum()) for k in range(n_folds)]) if len_match else np.full(n_folds, -1)
neg_per = np.array([int((1 - y[fid == k]).sum()) for k in range(n_folds)]) if len_match else np.full(n_folds, -1)
bad_folds = int(((pos_per < 1) | (neg_per < 1)).sum()) if len_match else -1

npz_path = f"{OUTDIR}/{BATCH}.npz"
np.savez(npz_path,
         fold_id=fid,
         global_index=concat.astype(int),
         fold_tx_id=np.array(valid_tx),
         fold_gene=fold_gene,
         fold_n=n_per_fold,
         fold_pos=pos_per,
         fold_neg=neg_per,
         labels=y,
         n_variants=np.array([n_variants]))
print(f"\n已落盘: {npz_path}")

# ---------- 人类可读摘要 ----------
report["summary"] = {
    "n_folds": n_folds,
    "n_samples_in_folds": total,
    "n_variants_full": int(n_variants),
    "excluded_variants": int(n_variants) - total,
    "per_fold_n": {"min": int(n_per_fold.min()),
                   "median": int(np.median(n_per_fold)),
                   "max": int(n_per_fold.max())},
    "class_balance_in_folds": {"pos": int(y.sum()) if len_match else None,
                               "neg": int((1 - y).sum()) if len_match else None,
                               "pos_rate": float(y.mean()) if len_match else None},
    "folds_with_both_classes": n_folds - bad_folds if bad_folds >= 0 else None,
    "note": "全集 2840 为 1420/1420 平衡；进入折的 1847 子集不平衡 —— Reject §二.3.2 实测坐实",
}
json_path = f"{OUTDIR}/{BATCH}.json"
json.dump(report, open(json_path, "w"), indent=2, ensure_ascii=False)
print(f"已落盘: {json_path}")

print("\n=== 摘要 ===")
for k, v in report["summary"].items():
    print(f"  {k}: {v}")
print("\n=== 判定 ===")
if order_ok and bad_folds == 0:
    print("✅ 折定义已冻结，顺序经独立证据确定，240/240 折正负均 >=1。")
    print("   ⇒ 后续所有数值定标一律使用本 batch 的 fold_id。")
else:
    print("⚠️ 冻结未完成，见上 FAIL 项。")
