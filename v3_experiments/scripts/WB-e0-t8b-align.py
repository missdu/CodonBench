#!/usr/bin/env python
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E0-T8b 逐条对齐比对（纯 CPU）

T8 给的是"集合交集"，看不出**哪一半变了、为什么变**。
本脚本按位置逐条比对两批序列，并与链向交叉统计，
用来检验"只有负链那部分变了"这个假设是否成立。

判据（硬）：
  若假设成立 ⇒ 逐条相同的样本应几乎全是正链，逐条不同的几乎全是负链。
  若正链里也大量不同 ⇒ 说明差异不止链向，我的解释错了。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import json
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"


def load(p):
    return [str(x) for x in np.load(p, allow_pickle=True)]


def main():
    old_ref = load(OUT / "e1_seqs_ref.npy")
    new_ref = load(OUT / "e1s_ref.npy")
    old_alt = load(OUT / "e1_seqs_alt.npy")
    new_alt = load(OUT / "e1s_alt.npy")

    n = min(len(old_ref), len(new_ref))
    same_ref = np.array([old_ref[i] == new_ref[i] for i in range(n)])
    same_alt = np.array([old_alt[i] == new_alt[i] for i in range(n)])

    rep = {"n": int(n),
           "same_ref": int(same_ref.sum()),
           "same_alt": int(same_alt.sum()),
           "diff_ref": int(n - same_ref.sum()),
           "diff_alt": int(n - same_alt.sum()),
           "strand_cross": None}

    # 与链向交叉（链向文件与新批同源，逐条对齐）
    sp = OUT / "e1s_strand.npy"
    if sp.exists():
        st = np.load(sp)
        st = st[:n]
        vals = sorted(set(str(x) for x in st))
        cross = {}
        for v in vals:
            m = np.array([str(x) == v for x in st])
            cross[v] = {
                "n": int(m.sum()),
                "same_ref": int(same_ref[m].sum()),
                "diff_ref": int((~same_ref[m]).sum()),
                "same_ref_frac": round(float(same_ref[m].mean()), 4) if m.sum() else None,
            }
        rep["strand_cross"] = cross

    # 前 3 条不同的样本，看差异长什么样（长短？大小写？互补？）
    idx = np.where(~same_ref)[0][:3]
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    examples = []
    for i in idx:
        o, w = old_ref[i], new_ref[i]
        examples.append({
            "i": int(i),
            "len_old": len(o), "len_new": len(w),
            "old_head": o[:30], "new_head": w[:30],
            "is_upper_diff": (o.upper() == w.upper()),
            "is_revcomp": (o.translate(comp)[::-1] == w),
        })
    rep["examples"] = examples

    rp = OUT / "e0_t8b_align.json"
    with open(rp, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print("=" * 74)
    print("E0-T8b 逐条比对（n=%d）" % n)
    print("=" * 74)
    print("  ref 相同 %d (%.1f%%) / 不同 %d" % (rep["same_ref"], 100 * rep["same_ref"] / n, rep["diff_ref"]))
    print("  alt 相同 %d (%.1f%%) / 不同 %d" % (rep["same_alt"], 100 * rep["same_alt"] / n, rep["diff_alt"]))
    if rep["strand_cross"]:
        print("-" * 74)
        print("按链向交叉（same_ref 比例）：")
        for v, c in rep["strand_cross"].items():
            print("  strand=%-6s n=%-5d 相同 %-5d 不同 %-5d  %.1f%% 相同"
                  % (v, c["n"], c["same_ref"], c["diff_ref"], 100 * (c["same_ref_frac"] or 0)))
    print("-" * 74)
    for e in rep["examples"]:
        print("  i=%d len %d->%d  仅大小写差异=%s  反向互补=%s"
              % (e["i"], e["len_old"], e["len_new"], e["is_upper_diff"], e["is_revcomp"]))
        print("      old: %s" % e["old_head"])
        print("      new: %s" % e["new_head"])
    print("=" * 74)
    print("已写出:", rp)


if __name__ == "__main__":
    main()
