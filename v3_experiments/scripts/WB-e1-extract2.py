# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— 提嵌入（修正版）

关键修正（2026-10-02 排查所得）：
  * CodonBERT 的 tokenizer 词表是 **RNA 密码子（含 U）**，大小 69：
    [PAD][UNK][CLS][SEP][MASK] + 64 个 RNA 密码子（AAA, AAU, AAG ...）
  * 我们的序列是 **DNA（含 T）**，若直接分词，TTT / TCC 等全部落为 [UNK]
    → 实测：直接分词得到 [CLS, UNK, SEP]，**任何序列嵌入完全相同**
  * 且 `scripts/model_wrapper.py::_tokenize_codon` 的方式 1（直接 tokenizer(连续串)）
    不抛异常但产生 UNK，导致永远走不到方式 2/3 的密码子分词 —— 这是该封装的缺陷
  ⇒ 本脚本自行实现：**T→U 转换 + 按密码子分词 + mean pooling**
  ⇒ 实测 UNK = 0/32，且 ref/alt token 差异恰在变异密码子位置

输出：wb_rerun/e1v2_<model>_ref.npy / _alt.npy
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SUPP = BASE / "results" / "supplementary"
OUT = SUPP / "wb_rerun"
OUT.mkdir(exist_ok=True)

ref_seqs = list(np.load(OUT / "e1_seqs_ref.npy", allow_pickle=True))
alt_seqs = list(np.load(OUT / "e1_seqs_alt.npy", allow_pickle=True))
n = len(ref_seqs)
print("序列数: %d | ref/alt 不同: %d" % (
    n, sum(1 for a, b in zip(ref_seqs, alt_seqs) if a != b)))


def to_rna_codons(x):
    r = x.replace("T", "U")
    return " ".join([r[i:i + 3] for i in range(0, len(r) - len(r) % 3, 3)])


MODELS = [
    {"id": "codonbert", "hf_path": "lhallee/CodonBERT"},
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
report = {}

for mi in MODELS:
    mid = mi["id"]
    try:
        tok = AutoTokenizer.from_pretrained(mi["hf_path"], trust_remote_code=True)
        mdl = AutoModel.from_pretrained(mi["hf_path"], trust_remote_code=True).to(device)
        mdl.eval()
    except Exception as e:
        print("[%s] 加载失败: %s" % (mid, str(e)[:160]))
        continue
    print("\n== %s ==" % mid)

    # 先验证编码质量
    tr, ta = to_rna_codons(ref_seqs[0]), to_rna_codons(alt_seqs[0])
    ir, ia = tok(tr)["input_ids"], tok(ta)["input_ids"]
    unk = sum(1 for x in ir if x == tok.unk_token_id)
    diffpos = [i for i, (x, y) in enumerate(zip(ir, ia)) if x != y]
    print("  样例 token 数 %d | UNK %d | ref/alt 差异位置 %s" % (len(ir), unk, diffpos))

    def embed(seqs, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                batch = [to_rna_codons(s) for s in seqs[i:i + bs]]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
                h = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (h * mask).sum(1) / mask.sum(1)   # 按 attention_mask 平均
                out.append(emb.cpu().numpy())
        return np.concatenate(out, 0)

    Eref = embed(ref_seqs)
    Ealt = embed(alt_seqs)
    np.save(OUT / ("e1v2_%s_ref.npy" % mid), Eref)
    np.save(OUT / ("e1v2_%s_alt.npy" % mid), Ealt)
    d = float(np.abs(Eref - Ealt).mean())
    print("  ref %s alt %s | mean|ref-alt| = %.6g" % (Eref.shape, Ealt.shape, d))
    if d == 0:
        print("  ★ 警告：ref 与 alt 嵌入完全相同，模型未感知变异")
    report[mid] = {"shape": list(Eref.shape), "mean_abs_diff": d,
                   "unk_in_sample": unk, "diff_positions": diffpos}
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT / "e1v2_extract_report.json", "w") as f:
    json.dump(report, f, indent=1)
print("\n完成:", report)
