# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E0 全量执行（补做 10-02 欠下的那一步）

背景：WB-E0-必过测试.py 10-02 已建，但只跑过本地 ESM-2，
      "全量需服务器嵌入"之后再没执行 ⇒ P0-o 事故没被拦住。
本脚本把全量补上：遍历 wb_rerun 下所有嵌入，逐批跑 8 项卫生检查，
外加两条 10-03 事故暴露的新检查：
  * ref 与 alt 不得相同（14:14 bug 批就是 ref≡alt）
  * 同一模型的各臂行数必须一致

只读，不改任何文件。输出 wb_e0_all.out + e0_all_results.json
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

import json
import numpy as np
from pathlib import Path

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

lab_all = np.load(OUT / "e1s_labels.npy")
try:
    usable = np.load(OUT / "e1s_usable.npy").astype(bool)
except Exception:
    usable = None

results = {}


def check_one(path, expect_n, labels):
    """返回 dict(name, pass, detail)"""
    r = {"n_expected": expect_n, "checks": [], "pass": True}
    try:
        X = np.load(path, allow_pickle=False)
    except Exception as e:
        r["checks"].append(("loadable", False, str(e)[:60]))
        r["pass"] = False
        return r

    def rec(name, cond, detail):
        r["checks"].append((name, bool(cond), detail))
        if not cond:
            r["pass"] = False

    rec("2D shape", X.ndim == 2, str(X.shape))
    if X.ndim != 2:
        return r
    rec("rows == labels", X.shape[0] == labels.shape[0],
        "emb=%d lab=%d" % (X.shape[0], labels.shape[0]))
    rec("no NaN/Inf", bool(np.isfinite(X).all()),
        "NaN=%d Inf=%d" % (int(np.isnan(X).sum()), int(np.isinf(X).sum())))
    std = float(np.asarray(X, dtype=np.float64).std())
    rec("not constant", std > 1e-8, "std=%.6g" % std)
    if X.shape[0] == labels.shape[0]:
        u = np.unique(labels)
        rec("labels binary", set(u.tolist()) <= {0, 1}, "unique=%s" % (u[:6],))
        n1 = int((labels == 1).sum())
        ratio = n1 / max(1, len(labels))
        rec("balanced 1:1", abs(ratio - 0.5) <= 0.01,
            "pos=%d neg=%d ratio=%.4f" % (n1, len(labels) - n1, ratio))
    r["shape"] = list(X.shape)
    return r


print("=" * 78)
print("E0 全量执行 —— wb_rerun 下所有嵌入")
print("=" * 78)

# --- 1. 逐文件卫生检查 ---
files = sorted(OUT.glob("*.npy"))
# 非嵌入的辅助文件：序列字符串、掩码、标签、索引。跳过而非判 FAIL
skip = {"e1s_labels.npy", "e1s_usable.npy", "e1s_genes.npy",
        "e1s_variant_index.npy", "e1s_review_status.npy",
        "e1_seqs_ref.npy", "e1_seqs_alt.npy", "e1s_ref.npy", "e1s_alt.npy",
        "e1s_pseudo.npy", "e1s_phase_pick.npy", "e1s_strand.npy"}
print("\n[1] 逐文件卫生检查（%d 个 .npy，%d 个辅助文件跳过）"
      % (len(files), len(skip)))
for p in files:
    if p.name in skip:
        print("  %-34s %-9s 辅助文件" % (p.name, "SKIP"))
        continue
    if p.name.startswith("e1s_"):
        n_exp = int(usable.sum()) if usable is not None else 2682
        labels = lab_all[usable] if usable is not None else lab_all
    else:
        n_exp = len(lab_all)
        labels = lab_all
    r = check_one(p, n_exp, labels)
    results[p.name] = r
    flag = "PASS" if r["pass"] else "**FAIL**"
    bad = [c[0] for c in r["checks"] if not c[1]]
    print("  %-34s %-9s %s" % (p.name, flag, ("; ".join(bad) if bad else "ok")))

# --- 2. 新增：ref vs alt 不得相同 ---
print("\n[2] 新增检查：ref 与 alt 是否相同")
print("    判据分两类（10-03 修正，此前把恒等式误判成 bug）：")
print("      * pLM：同义不改蛋白 ⇒ ref/alt 是同一字符串 ⇒ **逐位相同是恒等式，属正确**")
print("      * 核酸模型：ref/alt 序列不同 ⇒ **逐位相同即为 bug 批**")
pairs = []
models = sorted({f.name[:-len("_ref.npy")] for f in OUT.glob("*_ref.npy")})
for m in models:
    is_plm = ("esm" in m.lower()) or ("prot" in m.lower()) or ("ankh" in m.lower())
    for a, b in [("ref", "alt"), ("ref", "pse"), ("ref", "pseudo")]:
        pa, pb = OUT / ("%s_%s.npy" % (m, a)), OUT / ("%s_%s.npy" % (m, b))
        if not (pa.exists() and pb.exists()):
            continue
        try:
            A, B = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
        except ValueError:
            print("  %-30s %-4s vs %-7s %s" % (m, a, b, "SKIP 非数值数组"))
            continue
        if A.dtype == object or B.dtype == object:
            print("  %-30s %-4s vs %-7s %s" % (m, a, b, "SKIP 非数值数组"))
            continue
        if A.shape != B.shape:
            print("  %-30s %s vs %s 形状不同 %s %s" % (m, a, b, A.shape, B.shape))
            continue
        d = float(np.abs(np.asarray(A, float) - np.asarray(B, float)).max())
        if d == 0.0:
            verdict = "IDENTICAL **恒等式(正确)**" if is_plm else "**IDENTICAL BUG**"
        else:
            verdict = "differs (ok)" if not is_plm else "differs **违反恒等式**"
        print("  %-30s %-4s vs %-7s maxdiff=%-10.4g %s"
              % (m, a, b, d, verdict))
        pairs.append((m, a, b, d, "plm" if is_plm else "na"))

# --- 3. 新增：同一模型各臂行数一致 ---
print("\n[3] 新增检查：同一模型各臂行数一致")
for m in models:
    arms = {}
    for arm in ["ref", "alt", "pse", "pseudo", "diff", "pair"]:
        p = OUT / ("%s_%s.npy" % (m, arm))
        if p.exists():
            try:
                arr = np.load(p, allow_pickle=False)
            except ValueError:
                continue
            if arr.dtype != object:
                arms[arm] = arr.shape[0]
    uniq = set(arms.values())
    flag = "ok" if len(uniq) <= 1 else "**INCONSISTENT**"
    print("  %-30s %-9s %s" % (m, flag, arms))

# --- 4. 汇总 ---
n_pass = sum(1 for r in results.values() if r["pass"])
n_fail = len(results) - n_pass
print("\n" + "=" * 78)
print("汇总：%d 个文件中 %d 通过 / %d 失败" % (len(results), n_pass, n_fail))
if n_fail:
    print("失败清单：")
    for k, r in results.items():
        if not r["pass"]:
            bad = [c[0] for c in r["checks"] if not c[1]]
            print("   - %s : %s" % (k, ", ".join(bad)))
identical = [p for p in pairs if p[3] == 0.0]
bug_batches = [p for p in identical if p[4] == "na"]
ok_identity = [p for p in identical if p[4] == "plm"]
if ok_identity:
    print("\n✅ pLM 恒等式成立（ref≡alt 属正确，不是 bug）：")
    for m, a, b, d, _ in ok_identity:
        print("   - %s : %s vs %s" % (m, a, b))
if bug_batches:
    print("\n🔴 核酸模型 ref≡alt 的 BUG 批次（任何用它的 Δ 都是垃圾）：")
    for m, a, b, d, _ in bug_batches:
        print("   - %s : %s vs %s" % (m, a, b))
if not bug_batches:
    print("\n✅ 无核酸模型的 ref≡alt bug 批次")
print("=" * 78)

json.dump({"files": results, "ref_alt_pairs": pairs},
          open(OUT / "e0_all_results.json", "w"), indent=1, default=str)
print("\n已写出 e0_all_results.json")
