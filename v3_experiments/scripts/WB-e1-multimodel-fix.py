# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— 多模型扩展 修复版（EXP-007b）

【修什么】首轮 WB-e1-multimodel.py 有 9 个模型加载或分词失败。本脚本逐个修：

| 模型 | 首轮失败原因 | 修法 |
|---|---|---|
| NT-v2-500m/50m | `MaskedLMOutput` 无 `last_hidden_state` | 请求 `output_hidden_states=True`，取 `hidden_states[-1]` |
| Mistral-Codon-117M/16M | `MixtralConfig` 不被 `AutoModelForMaskedLM` 接受 | 追加 `AutoModelForCausalLM` 兜底 |
| mRNABERT | 缺 `bert_padding` 包 | 改用项目内 `models/mrnabert` 本地目录 |
| cdsBERT / cdsBERT-plus | 词表不适配，UNK=24，嵌入退化为常数 | 追加 `nt_space`（逐核苷酸空格）等策略 |
| CodonTransformer | 同上 | 同上 |

【保留的硬约束（首轮已验证，不得放宽）】
  1. 核酸模型：mean|ref-alt| 必须 > 1e-6（模型必须感知变异）
  2. 蛋白模型：mean|ref-alt| 必须 < 1e-5（同义不改变蛋白 ⇒ 恒等式）
  3. 分词策略自动搜索，选 UNK 最低者，并记录所选策略
  4. 服务器无外网 ⇒ 必须 HF_HUB_OFFLINE=1

【只跑失败模型，已成功的（codonbert / esm1b-650m / esm2-8m）不重跑】
结果增量合并进 e1_multimodel_results.json。
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import gc
import numpy as np
import torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModel, AutoModelForMaskedLM,
                          AutoModelForCausalLM)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

_u = np.load(OUT / "e1s_usable.npy").astype(bool)
ref_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_ref.npy", allow_pickle=True)) if _u[i]]
alt_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_alt.npy", allow_pickle=True)) if _u[i]]
pse_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_pseudo.npy", allow_pickle=True)) if _u[i]]
_vix = np.load(OUT / "e1s_variant_index.npy")[_u]
_ph = np.load(OUT / "e1s_phase_pick.npy")[_u]
labels = np.load(OUT / "e1s_labels.npy")[_u].astype(int)
genes = np.array([g for i, g in enumerate(np.load(OUT / "e1s_genes.npy", allow_pickle=True)) if _u[i]],
                 dtype=object)
n = len(labels)
print("可用样本 %d | 标签 %s | 基因 %d" % (n, np.bincount(labels), len(set(genes))), flush=True)

_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {}
_k = 0
for x in _b:
    for y in _b:
        for z in _b:
            CODON[x + y + z] = _aa[_k]
            _k += 1


def to_prot(s, ix, ph):
    st = max(0, ix - ph)
    t = s[st:]
    L = len(t) - len(t) % 3
    return "".join(CODON.get(t[i:i + 3], "X") for i in range(0, L, 3))


prot_ref = [to_prot(s, _vix[i], _ph[i]) for i, s in enumerate(ref_seqs)]
prot_alt = [to_prot(s, _vix[i], _ph[i]) for i, s in enumerate(alt_seqs)]
prot_pse = [to_prot(s, _vix[i], _ph[i]) for i, s in enumerate(pse_seqs)]
_same = sum(1 for a, b in zip(prot_ref, prot_alt) if a == b)
assert _same == n, "同义变异的蛋白必须相同（%d/%d）" % (_same, n)
print("[构造自检] ref 蛋白 == alt 蛋白: %d / %d" % (_same, n), flush=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("设备:", device, flush=True)


def _codons(x):
    return " ".join(x[i:i + 3] for i in range(0, len(x) - len(x) % 3, 3))


def _nt_space(x):
    return " ".join(list(x))


STRATEGIES = {
    "raw": lambda s: s,
    "upper": lambda s: s.upper(),
    "rna": lambda s: s.upper().replace("T", "U"),
    "rna_codon": lambda s: _codons(s.upper().replace("T", "U")),
    "dna_codon": lambda s: _codons(s.upper()),
    "nt_space": lambda s: _nt_space(s.upper()),
    "nt_space_rna": lambda s: _nt_space(s.upper().replace("T", "U")),
    "nt_space_lower": lambda s: _nt_space(s.lower()),
}


def pick_strategy(tok, seqs, sample=24):
    if tok.unk_token_id is None:
        return "raw", 0
    sub = seqs[:sample]
    best, best_unk = "raw", None
    for name, fn in STRATEGIES.items():
        try:
            tot = sum(sum(1 for t in tok(fn(s))["input_ids"] if t == tok.unk_token_id) for s in sub)
            if best_unk is None or tot < best_unk:
                best, best_unk = name, tot
        except Exception:
            continue
    return best, int(best_unk or 0)


def load_any(path):
    """依次尝试 AutoModel / MaskedLM / CausalLM，返回 (model, 加载方式)"""
    for cls, tag in [(AutoModel, "AutoModel"),
                     (AutoModelForMaskedLM, "MaskedLM"),
                     (AutoModelForCausalLM, "CausalLM")]:
        try:
            return cls.from_pretrained(path, trust_remote_code=True,
                                       output_hidden_states=True), tag
        except Exception:
            continue
    raise RuntimeError("三种方式均加载失败: %s" % path)


@torch.no_grad()
def embed(tok, mdl, seqs, kind, strategy, bs=32):
    fn = STRATEGIES[strategy]
    out = []
    for i in range(0, len(seqs), bs):
        batch = [fn(s) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        o = mdl(**enc)
        h = getattr(o, "last_hidden_state", None)
        if h is None:                       # MaskedLMOutput 等没有该属性
            hs = getattr(o, "hidden_states", None)
            if hs is None:
                raise RuntimeError("输出既无 last_hidden_state 也无 hidden_states")
            h = hs[-1]
        m = enc["attention_mask"].unsqueeze(-1).float()
        if kind == "plm":
            m[:, 0] = 0.0
            idx = enc["attention_mask"].sum(1) - 1
            for r, j in enumerate(idx):
                m[r, int(j)] = 0.0
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0)


def auc_ci(y, s, groups, n_boot=1000, seed=7):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y[idx], s[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def oof_scores(X, y, groups, split):
    X = np.asarray(X, dtype=np.float64)
    oof = np.zeros(len(y))
    folds = (list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)) if split == "random"
             else list(GroupKFold(5).split(X, y, groups)))
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run_arms(Xdict, y, groups):
    res = {}
    for split in ["random", "LOGO"]:
        for arm, X in Xdict.items():
            a, lo, hi = auc_ci(y, oof_scores(X, y, groups, split), groups)
            res["%s_%s" % (arm, split)] = {
                "auc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return res


def paired_delta(y, groups, Xa, Xb, split, n_boot=1000, seed=11):
    oa = oof_scores(Xa, y, groups, split)
    ob = oof_scores(Xb, y, groups, split)
    da, db = roc_auc_score(y, oa), roc_auc_score(y, ob)
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y[idx], oa[idx]) - roc_auc_score(y[idx], ob[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc_A": round(float(da), 4), "auc_B": round(float(db), 4),
            "delta": round(float(da - db), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "excludes_zero": bool(lo > 0 or hi < 0)}


MODELS = [
    {"id": "nt-v2-500m", "path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
     "kind": "clm", "bs": 16},
    {"id": "nt-v2-50m", "path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
     "kind": "clm", "bs": 32},
    {"id": "mistral-codon-117m", "path": "RaphaelMourad/Mistral-Codon-v1-117M",
     "kind": "clm", "bs": 32},
    {"id": "mistral-codon-16m", "path": "RaphaelMourad/Mistral-Codon-v1-16M",
     "kind": "clm", "bs": 32},
    {"id": "mrnabert", "path": str(BASE / "models" / "mrnabert"), "kind": "clm", "bs": 32},
    {"id": "cdsbert", "path": "GleghornLab/cdsBERT", "kind": "clm", "bs": 32},
    {"id": "cdsbert-plus", "path": "GleghornLab/cdsBERT-plus", "kind": "clm", "bs": 32},
    {"id": "codontransformer", "path": "adibvafa/CodonTransformer", "kind": "clm", "bs": 32},
    {"id": "codonbert_hf", "path": "Taykhoom/mRNABERT", "kind": "clm", "bs": 32},
]

res_path = OUT / "e1_multimodel_results.json"
try:
    all_res = json.load(open(res_path, encoding="utf-8"))
except Exception:
    all_res = {}

for mi in MODELS:
    mid = mi["id"]
    print("\n========== %s (%s) ==========" % (mid, mi["kind"]), flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(mi["path"], trust_remote_code=True)
        mdl, how = load_any(mi["path"])
        mdl = mdl.to(device).eval()
        print("  加载方式: %s" % how, flush=True)
    except Exception as e:
        print("  加载失败，跳过: %s" % str(e)[:160], flush=True)
        all_res[mid] = {"status": "load_failed", "error": str(e)[:300]}
        with open(res_path, "w") as f:
            json.dump(all_res, f, indent=1, ensure_ascii=False)
        continue

    try:
        seqs3 = ([prot_ref, prot_alt, prot_pse] if mi["kind"] == "plm"
                 else [ref_seqs, alt_seqs, pse_seqs])
        strategy, unk = pick_strategy(tok, seqs3[0])
        print("  分词策略: %s | 样例 UNK: %d" % (strategy, unk), flush=True)

        E_ref = embed(tok, mdl, seqs3[0], mi["kind"], strategy, mi.get("bs", 32))
        E_alt = embed(tok, mdl, seqs3[1], mi["kind"], strategy, mi.get("bs", 32))
        E_pse = embed(tok, mdl, seqs3[2], mi["kind"], strategy, mi.get("bs", 32))
        d = float(np.abs(E_ref - E_alt).mean())

        ok = (d > 1e-6) if mi["kind"] == "clm" else (d < 1e-5)
        if mi["kind"] == "clm":
            print("  [自检 阳性] 核酸模型要求 mean|ref-alt| > 1e-6 ⇒ %.3e %s"
                  % (d, "通过" if ok else "★失败：模型未感知变异"), flush=True)
        else:
            print("  [自检 阴性] 蛋白模型要求 mean|ref-alt| < 1e-5 ⇒ %.3e %s"
                  % (d, "通过" if ok else "★失败"), flush=True)
        if not ok:
            all_res[mid] = {"status": "selfcheck_failed", "mean_abs_ref_alt": d,
                            "strategy": strategy, "unk": unk}
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
            with open(res_path, "w") as f:
                json.dump(all_res, f, indent=1, ensure_ascii=False)
            continue

        np.save(OUT / ("e1s_%s_ref.npy" % mid), E_ref)
        np.save(OUT / ("e1s_%s_alt.npy" % mid), E_alt)

        Xdict = {"ref": E_ref, "alt": E_alt, "diff": E_alt - E_ref,
                 "pair": np.concatenate([E_ref, E_alt], 1),
                 "pseudo_diff": E_pse - E_ref}
        r = run_arms(Xdict, labels, genes)
        for k in sorted(r):
            v = r[k]
            print("  %-22s AUC %.4f  [%.4f, %.4f]" % (k, v["auc"], v["ci_lo"], v["ci_hi"]), flush=True)
        pd_r = {}
        for split in ["random", "LOGO"]:
            pd_r[split] = paired_delta(labels, genes, E_alt - E_ref, E_pse - E_ref, split)
            v = pd_r[split]
            print("  配对 Δ(diff−pseudo) %-6s %+.4f  [%.4f, %.4f] %s"
                  % (split, v["delta"], v["ci_lo"], v["ci_hi"],
                     "CI 不含 0" if v["excludes_zero"] else "含 0"), flush=True)
        r["_paired"] = pd_r
        r["_selfcheck"] = {"mean_abs_ref_alt": d, "unk": int(unk), "strategy": strategy,
                           "kind": mi["kind"], "load": how, "status": "ok"}
        all_res[mid] = r
        with open(res_path, "w") as f:
            json.dump(all_res, f, indent=1, ensure_ascii=False)
        print("  [已落盘]", flush=True)
    except Exception as e:
        print("  运行失败: %s" % str(e)[:300], flush=True)
        all_res[mid] = {"status": "run_failed", "error": str(e)[:300]}
        with open(res_path, "w") as f:
            json.dump(all_res, f, indent=1, ensure_ascii=False)

    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(res_path, "w") as f:
    json.dump(all_res, f, indent=1, ensure_ascii=False)
print("\n全部完成:", res_path)
