# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— 第 1 步：提取 reference-only 与 alternate 嵌入

背景：CodeArts 的 E1 是"初步版"，用同义随机化嵌入近似 reference-only。
本脚本按设计书要求，构造**真实的** ref / alt 序列对并重新提嵌入。

序列构造（已验证，见实验日志）：
  * SynPath 的 `char_seq` 是以变异位点为中心的上下文窗口（46-91 bp）
  * `cpos` == HGVS `c.` 位置（2840/2840 命中）
  * 变异位点在 char_seq 索引 **45**，其中 2816/2840 (99.2%) 的字符 == 参考碱基
  ⇒ ref 序列 = char_seq 原样
  ⇒ alt 序列 = char_seq，第 45 位替换为 HGVS 解析出的替代碱基

模型：先用两个 cLM（CodonBERT、EnCodon-80M）。
      pLM（ESM）需翻译为氨基酸；同义变异不改变氨基酸 ⇒ ref/alt 蛋白序列相同，
      这是要验证的现象本身，单独处理。

输出：wb_rerun/e1_emb_<model>_ref.npy 与 _alt.npy
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
SYN = BASE / "results" / "unified_eval" / "synpath_data"
OUT = SUPP / "wb_rerun"
OUT.mkdir(exist_ok=True)

# 序列优先复用已保存版本（copra_h 环境无 pyarrow，读不了 parquet）
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_seq_r = OUT / "e1_seqs_ref.npy"
_seq_a = OUT / "e1_seqs_alt.npy"
if _seq_r.exists() and _seq_a.exists():
    ref_seqs = list(np.load(_seq_r, allow_pickle=True))
    alt_seqs = list(np.load(_seq_a, allow_pickle=True))
    n = len(ref_seqs)
    print("复用已保存序列: %d（跳过 parquet 与 HGVS 解析）" % n)
    print("  ref/alt 不同的样本数: %d" % sum(
        1 for a, b in zip(ref_seqs, alt_seqs) if a != b))
    df = None
else:
    df = pd.read_parquet(SYN / "synpath_variants.parquet")
    n = len(df)

# ---------- 1. 解析 HGVS，构造 ref / alt 序列 ----------
PAT = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
CENTER = 45

if df is not None:
    ref_seqs, alt_seqs = [], []
    match_ref = 0
    parsed = 0
    for i in range(n):
        row = df.iloc[i]
        m = PAT.search(str(row["Name"]))
        seq = row["char_seq"]
        if not m or len(seq) <= CENTER:
            ref_seqs.append(seq); alt_seqs.append(seq); continue
        parsed += 1
        _pos, rbase, abase = m.groups()
        if seq[CENTER] == rbase:
            match_ref += 1
        ref_seqs.append(seq)
        alt_seqs.append(seq[:CENTER] + abase + seq[CENTER + 1:])

    print("解析 HGVS 成功: %d / %d" % (parsed, n))
    print("char_seq[%d] == 参考碱基: %d / %d (%.1f%%)" % (
        CENTER, match_ref, n, 100.0 * match_ref / n))
    diff = sum(1 for a, b in zip(ref_seqs, alt_seqs) if a != b)
    print("ref 与 alt 序列不同的样本数: %d / %d" % (diff, n))

    np.save(OUT / "e1_seqs_ref.npy", np.array(ref_seqs, dtype=object), allow_pickle=True)
    np.save(OUT / "e1_seqs_alt.npy", np.array(alt_seqs, dtype=object), allow_pickle=True)
    with open(OUT / "e1_seq_meta.json", "w") as f:
        json.dump({"n": n, "parsed_hgvs": parsed, "center_index": CENTER,
                   "ref_base_match": match_ref, "n_ref_alt_differ": diff,
                   "note": "ref=char_seq; alt=char_seq with index 45 replaced by HGVS alt base"},
                  f, indent=1)

# ---------- 2. 模型 ----------
import sys
sys.path.insert(0, str(BASE / "scripts"))
from model_wrapper import create_wrapper  # noqa: E402

MODELS = [
    {"id": "codonbert", "hf_path": "lhallee/CodonBERT"},
    {"id": "encodon-80m", "hf_path": "goodarzilab/encodon-80M"},
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("\n使用设备:", device)

for mi in MODELS:
    mid = mi["id"]
    try:
        w = create_wrapper(mi, device=device)
    except Exception as e:
        print("  [%s] 加载失败: %s" % (mid, str(e)[:150]))
        continue
    print("  提取 %s ..." % mid)
    try:
        Eref = np.asarray(w.get_embeddings(ref_seqs, batch_size=16))
        Ealt = np.asarray(w.get_embeddings(alt_seqs, batch_size=16))
    except Exception as e:
        print("  [%s] 提嵌入失败: %s" % (mid, str(e)[:200]))
        continue
    np.save(OUT / ("e1_emb_%s_ref.npy" % mid), Eref)
    np.save(OUT / ("e1_emb_%s_alt.npy" % mid), Ealt)
    print("    ref %s  alt %s  已保存" % (Eref.shape, Ealt.shape))
    # 一致性检查：ref 与 alt 嵌入的平均差异（应非零，否则说明模型没看到变异）
    d = np.abs(Eref - Ealt).mean()
    print("    mean|ref-alt| = %.6g (若为 0 说明模型未感知变异)" % d)
    del w
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n完成。产物目录:", OUT)
