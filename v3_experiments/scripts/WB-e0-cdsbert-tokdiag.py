#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-cdsbert-tokdiag.py —— cdsBERT / cdsBERT-plus 分词诊断

MODEL_REGISTRY 抓到：cdsBERT 对真实序列的 UNK 率 = 32/32 = 100%，
且加特殊 token 后整条 91 nt 序列只产生 3 个 token（CLS + 1 + SEP）。
⇒ 它把整条序列当成了**一个未知词**。

Reject §五.1 说得很准：预训练权重要求**精确保持 token-ID 映射**，
不是词表集合相同即可。若分词方式不对，前面所有 cdsBERT 的结果都是无效的。

要回答：
  Q1 词表里到底有什么？（vocab 只有 69 ⇒ 大概率是密码子级词表）
  Q2 正确的输入格式是什么？试几种切分法，看哪种 UNK=0
  Q3 若找到正确格式，重算 UNK 率与推理 shape
  Q4 cdsBERT 与 cdsBERT-plus 是不是同一份权重？（registry 显示参数量完全相同）

判据：找到一种格式使 UNK=0 **且** token 数与序列长度/3 相当 ⇒ 该格式才是可用的。
"""
import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
import numpy as np

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
OUT = f"{BASE}/results/supplementary/wb_rerun/wb_frozen"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
SEQ = [str(s) for s in df["char_seq"].values[:8]]
print("样本序列（前 60 nt）:")
for s in SEQ[:3]:
    print("  ", s[:60], f"(len={len(s)})")

from transformers import AutoTokenizer

result = {}
for mid, path in (("cdsbert", "GleghornLab/cdsBERT"),
                  ("cdsbert-plus", "GleghornLab/cdsBERT-plus")):
    print(f"\n================ {mid} ================")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                        trust_remote_code=True)
    rec = {"class": type(tok).__name__, "vocab_size": tok.vocab_size,
           "unk_token": str(tok.unk_token), "unk_id": tok.unk_token_id}
    print(f"  tokenizer class={rec['class']} vocab={rec['vocab_size']} "
          f"unk={rec['unk_token']}(id={rec['unk_id']})")

    # Q1 词表内容
    try:
        vt = tok.get_vocab()
        items = sorted(vt.items(), key=lambda kv: kv[1])[:80]
        rec["vocab_first80"] = [k for k, _ in items]
        print(f"  词表前 80: {rec['vocab_first80'][:40]}")
    except Exception as e:
        rec["vocab_err"] = str(e)[:200]

    # Q2 试各种输入格式
    s0 = SEQ[0]
    cands = {
        "raw_upper": s0.upper(),
        "raw_lower": s0.lower(),
        "space_per_codon": " ".join(s0.upper()[i:i + 3] for i in range(0, len(s0), 3)),
        "space_per_char": " ".join(list(s0.upper())),
        "space_per_3char_lower": " ".join(s0.lower()[i:i + 3] for i in range(0, len(s0), 3)),
    }
    rec["formats"] = {}
    for name, s in cands.items():
        try:
            ids = tok(s, add_special_tokens=False)["input_ids"]
            u = tok.unk_token_id
            n_unk = sum(1 for x in ids if x == u)
            rec["formats"][name] = {"n_tokens": len(ids), "n_unk": n_unk,
                                    "ids_head8": ids[:8]}
            print(f"  {name:22s} n_tokens={len(ids):5d} n_unk={n_unk:5d} "
                  f"head={ids[:8]}")
        except Exception as e:
            rec["formats"][name] = {"error": str(e)[:150]}
            print(f"  {name:22s} ERR {str(e)[:80]}")

    # Q3 若找到 UNK=0 的格式，重算 UNK 率
    best = None
    for name, r in rec["formats"].items():
        if isinstance(r, dict) and r.get("n_unk") == 0 and r.get("n_tokens", 0) > 1:
            best = name
            break
    rec["best_format"] = best
    if best:
        fmt = {"space_per_codon": lambda s: " ".join(
            s.upper()[i:i + 3] for i in range(0, len(s), 3)),
            "space_per_char": lambda s: " ".join(list(s.upper())),
            "raw_upper": lambda s: s.upper()}.get(best)
        if fmt:
            tot = unk = 0
            for s in SEQ:
                ids = tok(fmt(s), add_special_tokens=False)["input_ids"]
                tot += len(ids)
                unk += sum(1 for x in ids if x == tok.unk_token_id)
            rec["unk_after_fix"] = {"n_tokens": tot, "n_unk": unk,
                                    "rate": unk / tot if tot else None}
            print(f"  ✅ 修正格式 {best}：UNK {unk}/{tot}")
    result[mid] = rec

# Q4 两个模型是否同一份权重
try:
    import torch
    from transformers import AutoModel
    a = AutoModel.from_pretrained("GleghornLab/cdsBERT", local_files_only=True,
                                  trust_remote_code=True)
    b = AutoModel.from_pretrained("GleghornLab/cdsBERT-plus", local_files_only=True,
                                  trust_remote_code=True)
    sa, sb = a.state_dict(), b.state_dict()
    same_keys = set(sa.keys()) == set(sb.keys())
    diff = []
    for k in list(sa.keys())[:50]:
        if k in sb and sa[k].shape == sb[k].shape:
            diff.append(float((sa[k].float() - sb[k].float()).abs().max()))
    result["_same_weights"] = {
        "same_keys": same_keys,
        "n_compared": len(diff),
        "max_abs_diff_first50": max(diff) if diff else None,
        "verdict": ("IDENTICAL" if diff and max(diff) == 0 else
                    "DIFFERENT" if diff and max(diff) > 0 else "UNKNOWN"),
    }
    print(f"\n[Q4] cdsBERT vs cdsBERT-plus：same_keys={same_keys} "
          f"前50个张量最大差={max(diff) if diff else 'NA'} ⇒ "
          f"{result['_same_weights']['verdict']}")
except Exception as e:
    result["_same_weights"] = {"error": str(e)[:300]}
    print("[Q4] 比较失败:", str(e)[:200])

json.dump(result, open(f"{OUT}/cdsbert_tokdiag_v1.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\n[out] {OUT}/cdsbert_tokdiag_v1.json")
