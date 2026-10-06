#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-model-registry.py (2026-10-03)

Reject §五.1「模型身份与权重记录不一致」点名的具体矛盾：
  - S1 说 CodonBERT 约 10M；S35 说约 110M；S19 说 12 层 87M
  - 加载表写 ESM-1b-650M，却指向 esm1b_t12_85M
  - CodonBERT 本地 checkpoint 配"按训练数据手建 vocab"——预训练权重要求
    **精确保持 token-ID 映射**，不是词表集合相同即可

要求产出：每个 checkpoint 的准确标识、commit/hash、实际参数量、config、
tokenizer hash、token-ID 例子、未知 token 率、最小推理测试。

🔴 本脚本的一条硬约束：服务器**没有外网**（实测 huggingface.co 连不上，
   上一轮 encodon-80m 就因此 "tokenizer_load_failed"）。
   所以一律用 local_files_only=True；**拿不到的就写 UNAVAILABLE，不编造**。

判据（能证伪）：
  P1 参数量必须**数出来**（sum(p.numel())），不能抄论文自称
  P2 每个模型要有 token→id 的真实例子（用来验证 vocab 映射没被重排）
  P3 UNK 率要报；UNK>0 的模型必须在稿件里说明
  P4 最小推理：一条真实序列进去，出 (1, L, H) 的张量；shape 要写下来

输出：results/supplementary/wb_rerun/wb_frozen/model_registry_v1.json
"""
import os
import json
import hashlib
import traceback
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # 不碰别人的 GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
OUT = f"{BASE}/results/supplementary/wb_rerun/wb_frozen"
BATCH = "model_registry_v1"

MODELS = [
    {"id": "codonbert", "path": "lhallee/CodonBERT", "kind": "clm"},
    {"id": "encodon-80m", "path": "goodarzilab/encodon-80m", "kind": "clm"},
    {"id": "encodon-620m", "path": "goodarzilab/encodon-620m", "kind": "clm"},
    {"id": "codontransformer", "path": "adibvafa/CodonTransformer", "kind": "clm"},
    {"id": "mrnabert", "path": "YYLY66/mRNABERT", "kind": "clm"},
    {"id": "mistral-codon-117m", "path": "RaphaelMourad/Mistral-Codon-v1-117M", "kind": "clm"},
    {"id": "mistral-codon-16m", "path": "RaphaelMourad/Mistral-Codon-v1-16M", "kind": "clm"},
    {"id": "decodon-200m", "path": "goodarzilab/decodon-200M", "kind": "clm"},
    {"id": "cdsbert", "path": "GleghornLab/cdsBERT", "kind": "clm"},
    {"id": "cdsbert-plus", "path": "GleghornLab/cdsBERT-plus", "kind": "clm"},
    {"id": "nt-v2-500m", "path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "kind": "clm"},
    {"id": "nt-v2-50m", "path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "kind": "clm"},
    {"id": "esm1b-650m", "path": "facebook/esm1b_t33_650M_UR50S", "kind": "plm"},
    {"id": "esm2-650m", "path": "facebook/esm2_t33_650M_UR50D", "kind": "plm"},
    {"id": "esm2-8m", "path": "facebook/esm2_t6_8M_UR50D", "kind": "plm"},
    {"id": "prot_t5", "path": "Rostlab/prot_t5_xl_half_uniref50-enc", "kind": "plm"},
]

# 真实序列，用来做 UNK 率与最小推理（取数据里的前 32 条）
df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
SEQ = [str(s) for s in df["char_seq"].values[:32]]
print(f"[data] 取 {len(SEQ)} 条真实序列做 UNK 率与最小推理", flush=True)


def sha256_head(path, nbytes=64 * 1024 * 1024):
    """大文件只算前 64MB，够用于身份识别且不会跑很久。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read(nbytes))
        return h.hexdigest()[:32]
    except Exception:
        return None


def weight_files(d):
    out = {}
    for fn in ("pytorch_model.bin", "model.safetensors", "tf_model.h5"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            out[fn] = {"bytes": os.path.getsize(p), "sha256_head64MB": sha256_head(p)}
    return out


reg = {}
for mi in MODELS:
    mid, path = mi["id"], mi["path"]
    rec = {"id": mid, "hf_repo_id": path, "kind": mi["kind"],
           "status": None, "error": None}
    print(f"\n=== {mid} ({path}) ===", flush=True)
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForMaskedLM
        import torch

        # --- config ---
        cfg = AutoConfig.from_pretrained(path, local_files_only=True,
                                         trust_remote_code=True)
        rec["config"] = {k: getattr(cfg, k, None) for k in
                         ("model_type", "architectures", "num_hidden_layers",
                          "hidden_size", "intermediate_size", "num_attention_heads",
                          "vocab_size", "max_position_embeddings")}
        rec["config_path_resolved"] = type(cfg).__name__

        # --- tokenizer ---
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                            trust_remote_code=True)
        rec["tokenizer"] = {
            "class": type(tok).__name__,
            "vocab_size": int(getattr(tok, "vocab_size", -1)),
            "unk_token": str(getattr(tok, "unk_token", None)),
            "unk_token_id": getattr(tok, "unk_token_id", None),
            "pad_token": str(getattr(tok, "pad_token", None)),
            "model_max_length": getattr(tok, "model_max_length", None),
        }
        # P2：token→id 真实例子（用真实序列，不是手造字符串）
        ex = tok(SEQ[0], add_special_tokens=False)
        ids = list(map(int, ex["input_ids"]))[:12]
        rec["token_id_examples"] = {
            "seq_head": SEQ[0][:40],
            "ids_head12": ids,
            "tokens_head12": [str(tok.convert_ids_to_tokens(i)) for i in ids],
        }
        # P3：UNK 率
        tot = unk = 0
        for s in SEQ:
            ii = tok(s, add_special_tokens=False)["input_ids"]
            tot += len(ii)
            u = rec["tokenizer"]["unk_token_id"]
            if u is not None:
                unk += sum(1 for x in ii if x == u)
        rec["unk_rate"] = {"n_tokens": tot, "n_unk": unk,
                           "rate": (unk / tot) if tot else None}

        # --- 模型本体 + P1 参数量 ---
        try:
            mdl = AutoModel.from_pretrained(path, local_files_only=True,
                                            trust_remote_code=True)
        except Exception:
            mdl = AutoModelForMaskedLM.from_pretrained(path, local_files_only=True,
                                                       trust_remote_code=True)
        n_params = int(sum(p.numel() for p in mdl.parameters()))
        rec["params"] = {"total": n_params,
                         "total_M": round(n_params / 1e6, 2),
                         "trainable_M": round(
                             sum(p.numel() for p in mdl.parameters()
                                 if p.requires_grad) / 1e6, 2)}
        rec["hidden_size_from_tensor"] = int(getattr(mdl.config, "hidden_size", -1))

        # --- P4 最小推理 ---
        mdl.eval()
        with torch.no_grad():
            enc = tok(SEQ[0], return_tensors="pt", truncation=True, max_length=128)
            out = mdl(**{k: v for k, v in enc.items()
                         if k in ("input_ids", "attention_mask")})
            last = out.last_hidden_state
        rec["min_inference"] = {"shape": list(last.shape),
                                "dtype": str(last.dtype),
                                "finite": bool(torch.isfinite(last).all())}

        # --- 权重文件身份 ---
        try:
            cache_dir = os.path.dirname(
                os.path.dirname(mdl.name_or_path)) if os.path.isdir(
                str(mdl.name_or_path)) else None
        except Exception:
            cache_dir = None
        rec["weight_files"] = weight_files(str(mdl.name_or_path)) \
            if os.path.isdir(str(mdl.name_or_path)) else "not_a_local_dir"

        rec["status"] = "ok"
        print(f"  参数量 {rec['params']['total_M']} M；"
              f"层 {rec['config'].get('num_hidden_layers')}；"
              f"vocab {rec['tokenizer']['vocab_size']}；"
              f"UNK {rec['unk_rate']['n_unk']}/{rec['unk_rate']['n_tokens']}；"
              f"推理 shape {rec['min_inference']['shape']}", flush=True)
        del mdl
    except Exception as e:
        rec["status"] = "UNAVAILABLE"
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  ❌ {rec['error']}", flush=True)
    reg[mid] = rec

ok = sum(1 for v in reg.values() if v["status"] == "ok")
print(f"\n[summary] 可离线登记 {ok}/{len(MODELS)}；"
      f"不可用 {len(MODELS)-ok}", flush=True)

json.dump({"batch": BATCH,
           "offline_note": "服务器无外网；全部用 local_files_only=True；"
                           "拿不到即标 UNAVAILABLE，不编造",
           "models": reg,
           "summary": {"n_total": len(MODELS), "n_ok": ok}},
          open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"[out] {OUT}/{BATCH}.json", flush=True)
