# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | MisPath 错义通道 · 多模型响应向量（EXP-009）

【为什么补跑】
主图第五幕里 NT-v2 / cdsBERT 等模型的**错义通道**一格写着"待跑"。
成稿里不能有"待跑"，要么跑出来，要么明确写"未测"。这里跑出来。

【与 EXP-008 的关系】
EXP-008 只跑了 2 个 pLM + CodonBERT；本实验扩到全部可用模型（7 个），
口径与 EXP-008 逐位一致（同一 npz、同一 pseudo 位置、同一 bootstrap 种子）。

【自检方向（与 SynPath 相反）】
  MisPath 上 pLM 要求 mean|ref-alt| > 1e-6（阳性自检：错义必须改变蛋白）。

【⚠️ pseudo 对照的通道差异（决定结果怎么读）】
  MisPath 的 pseudo 在上游 15 nt 改一个碱基，它**也会改变蛋白**（9666/9999）
  ⇒ 对 pLM 不是干净对照 ⇒ pLM 只看 diff 的绝对水平，不报 diff−pseudo 增量。
  对核酸模型，pseudo 是"无关位置的真实序列变化"，是合适对照 ⇒ 报配对增量。

【不作的比较】
  SynPath 与 MisPath 窗口长度不同（30 vs 15 密码子）、数据集不同
  ⇒ 跨数据集的数值不可比大小，只作"同一模型在不同通道上的有无"比较。
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
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

_z = np.load(OUT / "mispath_seqs.npz", allow_pickle=True)
labels = _z["labels"].astype(int)
genes = _z["genes"].astype(object)
ref_nt = list(_z["ref_nt"])
alt_nt = list(_z["alt_nt"])
prot_ref = list(_z["prot_ref"])
prot_alt = list(_z["prot_alt"])
n = len(labels)
print("可用 %d | 标签 %s | 基因 %d" % (n, np.bincount(labels), len(set(genes))), flush=True)

_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {}
_k = 0
for x in _b:
    for y in _b:
        for z in _b:
            CODON[x + y + z] = _aa[_k]
            _k += 1

VARIANT_NT = 21
PSEUDO_NT = VARIANT_NT - 15
FLIP = {"A": "C", "C": "G", "G": "T", "T": "A"}


def make_pseudo(s):
    if len(s) <= PSEUDO_NT:
        return s
    c = s[PSEUDO_NT]
    return s[:PSEUDO_NT] + FLIP.get(c, "A") + s[PSEUDO_NT + 1:]


pse_nt = [make_pseudo(s) for s in ref_nt]
prot_pse = ["".join(CODON.get(s[i:i + 3], "X") for i in range(0, len(s) - len(s) % 3, 3))
            for s in pse_nt]
_d = sum(1 for a, b in zip(prot_ref, prot_pse) if a != b)
print("pseudo 也改变蛋白的: %d / %d ⇒ 对 pLM 不干净，pLM 不报配对增量" % (_d, n), flush=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("设备:", device, flush=True)


def _codons(x):
    return " ".join(x[i:i + 3] for i in range(0, len(x) - len(x) % 3, 3))


STRATEGIES = {
    "raw": lambda s: s,
    "upper": lambda s: s.upper(),
    "rna": lambda s: s.upper().replace("T", "U"),
    "rna_codon": lambda s: _codons(s.upper().replace("T", "U")),
    "dna_codon": lambda s: _codons(s.upper()),
    "nt_space": lambda s: " ".join(s.upper()),
    "aa_space": lambda s: " ".join(s.upper()),
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


@torch.no_grad()
def embed(tok, mdl, seqs, kind, strategy, bs=32):
    fn = STRATEGIES[strategy]
    out = []
    for i in range(0, len(seqs), bs):
        batch = [fn(s) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        o = mdl(**enc, output_hidden_states=True)
        h = getattr(o, "last_hidden_state", None)
        if h is None:
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


def paired_delta(y, groups, sA, sB, n_boot=1000, seed=7):
    """配对 bootstrap：同一批样本上两个表示的 AUC 差"""
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    gi = {g: np.where(groups == g)[0] for g in ug}
    a = roc_auc_score(y, sA)
    b = roc_auc_score(y, sB)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(ug), size=len(ug), replace=True)
        idx = np.concatenate([gi[ug[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            boots.append(roc_auc_score(y[idx], sA[idx]) - roc_auc_score(y[idx], sB[idx]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc_A": round(float(a), 4), "auc_B": round(float(b), 4),
            "delta": round(float(a - b), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)}


MODELS = [
    {"id": "esm1b-650m",    "path": "facebook/esm1b_t33_650M_UR50S", "kind": "plm", "bs": 16},
    {"id": "esm2-8m",       "path": "facebook/esm2_t6_8M_UR50D",     "kind": "plm", "bs": 64},
    {"id": "codonbert",     "path": "lhallee/CodonBERT",              "kind": "clm", "bs": 32},
    {"id": "cdsbert",       "path": "GleghornLab/cdsBERT",            "kind": "clm", "bs": 32},
    {"id": "cdsbert-plus",  "path": "GleghornLab/cdsBERT-plus",       "kind": "clm", "bs": 32},
    {"id": "nt-v2-500m",
     "path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "kind": "clm", "bs": 16},
    {"id": "nt-v2-50m",
     "path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",  "kind": "clm", "bs": 32},
]

all_res = {"n": n, "n_genes": int(len(set(genes))),
           "note": "EXP-009 MisPath 错义通道多模型；pLM 不报配对增量（pseudo 也改变蛋白）"}
for mi in MODELS:
    mid = mi["id"]
    print("\n========== %s (%s) ==========" % (mid, mi["kind"]), flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(mi["path"], trust_remote_code=True)
        try:
            mdl = AutoModel.from_pretrained(mi["path"], trust_remote_code=True)
        except Exception:
            try:
                from transformers import AutoModelForMaskedLM
                mdl = AutoModelForMaskedLM.from_pretrained(mi["path"], trust_remote_code=True)
            except Exception:
                from transformers import AutoModelForCausalLM
                mdl = AutoModelForCausalLM.from_pretrained(mi["path"], trust_remote_code=True)
        mdl = mdl.to(device).eval()
    except Exception as e:
        print("  加载失败，跳过: %s" % str(e)[:160], flush=True)
        all_res[mid] = {"status": "load_failed", "error": str(e)[:300]}
        continue

    try:
        seqs3 = ([prot_ref, prot_alt, prot_pse] if mi["kind"] == "plm"
                 else [ref_nt, alt_nt, pse_nt])
        strategy, unk = pick_strategy(tok, seqs3[0])
        print("  分词策略: %s | 样例 UNK: %d" % (strategy, unk), flush=True)

        E_ref = embed(tok, mdl, seqs3[0], mi["kind"], strategy, mi.get("bs", 32))
        E_alt = embed(tok, mdl, seqs3[1], mi["kind"], strategy, mi.get("bs", 32))
        E_pse = embed(tok, mdl, seqs3[2], mi["kind"], strategy, mi.get("bs", 32))
        d = float(np.abs(E_ref - E_alt).mean())
        print("  mean|ref-alt| = %.3e" % d, flush=True)
        if d <= 1e-6:
            print("  ★自检失败：错义未改变蛋白", flush=True)
            all_res[mid] = {"status": "selfcheck_failed", "mean_abs_ref_alt": d}
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
            continue
        print("  [自检 阳性] 通过（蛋白确实变了）", flush=True)

        Xdict = {"ref": E_ref, "alt": E_alt, "diff": E_alt - E_ref,
                 "pair": np.concatenate([E_ref, E_alt], 1),
                 "pseudo_diff": E_pse - E_ref}
        r = {}
        for split in ["random", "LOGO"]:
            for arm, X in Xdict.items():
                oof = oof_scores(X, labels, genes, split)
                a, lo, hi = auc_ci(labels, oof, genes)
                r["%s_%s" % (arm, split)] = {"auc": round(a, 4), "ci_lo": round(lo, 4),
                                             "ci_hi": round(hi, 4)}
                print("  %-22s AUC %.4f  [%.4f, %.4f]" % ("%s_%s" % (arm, split), a, lo, hi),
                      flush=True)
            if mi["kind"] == "clm":
                oA = oof_scores(Xdict["diff"], labels, genes, split)
                oB = oof_scores(Xdict["pseudo_diff"], labels, genes, split)
                pr = paired_delta(labels, genes, oA, oB)
                r.setdefault("_paired", {})[split] = pr
                print("  配对 Δ(diff−pseudo) %-6s %+.4f  [%.4f, %.4f] %s"
                      % (split, pr["delta"], pr["ci_lo"], pr["ci_hi"],
                         "CI 不含 0" if (pr["ci_lo"] > 0 or pr["ci_hi"] < 0) else "CI 含 0"),
                      flush=True)
        r["_selfcheck"] = {"mean_abs_ref_alt": d, "unk": int(unk), "strategy": strategy,
                           "kind": mi["kind"], "status": "ok"}
        all_res[mid] = r
        with open(OUT / "mispath_multimodel_results.json", "w") as f:
            json.dump(all_res, f, indent=1, ensure_ascii=False)
        print("  [已落盘]", flush=True)
    except Exception as e:
        print("  运行失败: %s" % str(e)[:300], flush=True)
        all_res[mid] = {"status": "run_failed", "error": str(e)[:300]}

    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT / "mispath_multimodel_results.json", "w") as f:
    json.dump(all_res, f, indent=1, ensure_ascii=False)
print("\n全部完成:", OUT / "mispath_multimodel_results.json")
