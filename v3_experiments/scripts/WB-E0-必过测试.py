# -*- coding: utf-8 -*-
"""
WB | E0 必过测试（最小够用版）

用途：在跑任何新实验之前，先验证数据/嵌入基线没有低级错误。
本项目背景：SI Table S36 曾出现"字段名带 _pp 却没乘 100"的单位错，
根因就是缺少这类自动检查。E0 是它的制度性防线。

用法：
  python WB-E0-必过测试.py <emb.npy> <labels.npy> [期望样本数]
  python WB-E0-必过测试.py --selftest     # 负例自检（故意喂坏数据，必须非 0 退出）

设计原则（与项目委派单纪律一致）：
  * 每个检查必须**能真报错**，不允许静默跳过；
  * 找不到数据就写"无法执行 + 缺什么"，不许猜；
  * 退出码：0=全过，2=有检查失败，3=负例自检失败（不该失败却通过了）。
"""

import sys
import os

try:
    import numpy as np
except ImportError:
    print("FAIL: numpy 不可用")
    sys.exit(2)

N_DEFAULT = 2840  # SynPath 已知规模


def check(emb_path, lab_path, expect_n=N_DEFAULT):
    """返回 (passed: bool, lines: list[str])"""
    lines = []
    ok = True

    def rec(name, cond, detail):
        nonlocal ok
        lines.append("  [%s] %-34s %s" % ("PASS" if cond else "FAIL", name, detail))
        if not cond:
            ok = False

    if not os.path.exists(emb_path):
        lines.append("  [FAIL] 嵌入文件不存在: %s" % emb_path)
        return False, lines
    if not os.path.exists(lab_path):
        lines.append("  [FAIL] 标签文件不存在: %s" % lab_path)
        return False, lines

    X = np.load(emb_path, allow_pickle=False)
    y = np.load(lab_path, allow_pickle=False)

    # 1 二维形状
    rec("嵌入是二维矩阵", X.ndim == 2, "shape=%s" % (X.shape,))
    # 2 标签一维
    rec("标签是一维", y.ndim == 1, "shape=%s" % (y.shape,))
    if X.ndim != 2 or y.ndim != 1:
        return ok, lines

    # 3 行数对齐（最常见也最致命）
    rec("嵌入行数 == 标签数", X.shape[0] == y.shape[0],
        "emb=%d vs lab=%d" % (X.shape[0], y.shape[0]))
    # 4 样本规模
    rec("样本数 == %d" % expect_n, X.shape[0] == expect_n, "实际 %d" % X.shape[0])
    # 5 无 NaN / Inf
    rec("嵌入无 NaN/Inf", bool(np.isfinite(X).all()),
        "NaN=%d Inf=%d" % (int(np.isnan(X).sum()), int(np.isinf(X).sum())))
    # 6 标签取值合法（二分类）
    u = np.unique(y)
    rec("标签为二分类 {0,1}", set(u.tolist()) <= {0, 1}, "unique=%s" % (u[:8],))
    # 7 类别平衡（SynPath 已知 1:1）
    if set(u.tolist()) <= {0, 1}:
        n1 = int((y == 1).sum())
        n0 = int((y == 0).sum())
        ratio = n1 / max(1, (n0 + n1))
        rec("类别 1:1 平衡（±1%）", abs(ratio - 0.5) <= 0.01,
            "pos=%d neg=%d ratio=%.4f" % (n1, n0, ratio))
    # 8 嵌入非常数（防止"提了个常数向量"却没人发现）
    if X.shape[0] > 1:
        std = float(X.std())
        rec("嵌入非常数向量", std > 1e-8, "std=%.6g" % std)

    return ok, lines


def selftest():
    """负例自检：故意构造坏数据，脚本必须报错（非 0 退出）。"""
    import tempfile
    print("=== 负例自检（这些输入都必须 FAIL）===")
    tmp = tempfile.mkdtemp(prefix="wb_e0_neg_")
    allbad = True

    def one(name, X, y, expect_n=N_DEFAULT):
        nonlocal allbad
        ep = os.path.join(tmp, "e.npy")
        lp = os.path.join(tmp, "l.npy")
        np.save(ep, X)
        np.save(lp, y)
        ok, lines = check(ep, lp, expect_n)
        print("-- %s" % name)
        for l in lines:
            print("   " + l)
        if ok:
            print("   ★ 自检失败：坏数据竟然全部通过了")
            allbad = False
        else:
            print("   ✓ 正确报错")
        return not ok

    n = 100
    good_X = np.random.RandomState(0).randn(n, 8)
    good_y = np.array([0, 1] * (n // 2))

    one("行数不对齐", good_X, good_y[:50])
    one("含 NaN", np.where(np.arange(n * 8).reshape(n, 8) == 5, np.nan, good_X), good_y)
    one("标签非二分类", good_X, np.random.RandomState(1).randint(0, 5, n))
    one("常数嵌入", np.zeros((n, 8)), good_y)
    one("样本数不符", good_X, good_y, expect_n=2840)

    print()
    if allbad:
        print("负例自检通过：所有坏输入都被拦下 ✓")
        return 0
    print("负例自检失败 ✗")
    return 3


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest())

    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    emb, lab = sys.argv[1], sys.argv[2]
    expect = int(sys.argv[3]) if len(sys.argv) >= 4 else N_DEFAULT

    print("=== E0 必过测试 ===")
    print("嵌入: %s" % emb)
    print("标签: %s" % lab)
    print("期望样本数: %d" % expect)
    ok, lines = check(emb, lab, expect)
    for l in lines:
        print(l)
    print()
    print("结果: %s" % ("全部通过" if ok else "有检查未通过"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
