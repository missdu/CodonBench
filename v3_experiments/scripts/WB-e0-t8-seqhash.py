#!/usr/bin/env python
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E0-T8 序列哈希（纯 CPU）

要回答一个问题：磁盘上几批嵌入，喂进去的序列到底是不是同一份？
没有哈希时，两批数值对不上，分不清是"序列变了"还是"提取方式变了"。
今天（10-03）的 P0-o 事故正是缺它。

输出：
  wb_rerun/e0_seqhash.json            逐文件摘要 + 两两重叠度
  wb_rerun/seqhash/<file>.sha256.npy  逐条哈希（供冻结数据包复用）
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("WB_BASE", os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")))
OUT = BASE / "results" / "supplementary" / "wb_rerun"
HDIR = OUT / "seqhash"
HDIR.mkdir(parents=True, exist_ok=True)

# 已知的序列来源文件（其余自动扫描补充）
KNOWN = [
    "e1_seqs_ref.npy", "e1_seqs_alt.npy",          # 旧批：extract.py 自己造（未定链）
    "e1s_ref.npy", "e1s_alt.npy", "e1s_pseudo.npy",  # 新批：phase A2（已定链定框）
]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_seqs(p: Path):
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        out = {}
        for k in d.files:
            a = d[k]
            if a.dtype == object or (a.ndim == 1 and a.dtype.kind in ("U", "S")):
                out["%s:%s" % (p.name, k)] = [str(x) for x in a]
        return out
    a = np.load(p, allow_pickle=True)
    if a.dtype == object or a.dtype.kind in ("U", "S"):
        return {p.name: [str(x) for x in a]}
    return {}


def main():
    files = []
    for f in KNOWN:
        if (OUT / f).exists():
            files.append(OUT / f)
    for p in sorted(OUT.glob("*.npy")):
        if p.name.startswith(("e1_seqs", "e1s_")) and "review" not in p.name \
                and "strand" not in p.name and "usable" not in p.name \
                and "index" not in p.name and "phase" not in p.name \
                and "labels" not in p.name and "genes" not in p.name:
            if p not in files:
                files.append(p)
    for p in sorted(OUT.glob("*.npz")):
        if p not in files:
            files.append(p)

    store = {}
    report = {"files": {}, "pairs": []}

    for p in files:
        try:
            seqs_by_key = load_seqs(p)
        except Exception as e:
            report["files"][p.name] = {"status": "LOAD_FAIL", "err": str(e)[:150]}
            continue
        if not seqs_by_key:
            report["files"][p.name] = {"status": "SKIP_NOT_SEQ"}
            continue
        for key, seqs in seqs_by_key.items():
            hs = [sha(s) for s in seqs]
            lens = [len(s) for s in seqs]
            np.save(HDIR / (key.replace(":", "__") + ".sha256.npy"),
                    np.array(hs, dtype=object), allow_pickle=True)
            store[key] = set(hs)
            report["files"][key] = {
                "status": "ok",
                "n": len(seqs),
                "len_min": min(lens), "len_max": max(lens),
                "len_modes": sorted(set(lens))[:6],
                "global_sha": sha("\n".join(hs))[:16],
                "head3": [s[:40] for s in seqs[:3]],
            }

    keys = sorted(store.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            inter = len(store[a] & store[b])
            denom = min(len(store[a]), len(store[b]))
            report["pairs"].append({
                "a": a, "b": b,
                "overlap": inter,
                "overlap_frac_of_smaller": round(inter / denom, 4) if denom else 0.0,
                "n_a": len(store[a]), "n_b": len(store[b]),
                "identical": (inter == len(store[a]) == len(store[b])),
            })

    rp = OUT / "e0_seqhash.json"
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("E0-T8 序列哈希（%d 个序列集）" % len(store))
    print("=" * 78)
    for k, v in report["files"].items():
        if v["status"] == "ok":
            print("  %-28s n=%-6d len=%d..%d  global=%s"
                  % (k, v["n"], v["len_min"], v["len_max"], v["global_sha"]))
        else:
            print("  %-28s %s" % (k, v["status"]))
    print("-" * 78)
    print("两两重叠（overlap / min(n_a,n_b)）：")
    for pr in report["pairs"]:
        flag = "IDENTICAL" if pr["identical"] else ("DISJOINT" if pr["overlap"] == 0 else "PARTIAL")
        print("  %-46s %6d  %6.2f%%  %s"
              % (pr["a"] + " ∩ " + pr["b"], pr["overlap"],
                 100 * pr["overlap_frac_of_smaller"], flag))
    print("=" * 78)
    print("已写出:", rp)


if __name__ == "__main__":
    main()
