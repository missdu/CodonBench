# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— 多模型扩展（EXP-007）

【为什么必须扩展】
EXP-004（E1 严格版）只跑了 2 个模型（CodonBERT + ESM-1b）。
V1 的面板是 21 个模型。只有 2 个模型的"响应向量"结论撑不起一篇论文。
本脚本把响应向量分析扩展到 V1 面板中本地可加载的全部模型。

【核心设计：响应向量】
  Δ = E(alt) - E(ref)        变异引入的表示变化（配对差分，背景在构造上消去）
  Δ_pseudo = E(pseudo) - E(ref)   同一窗口换个不相干碱基 ⇒ 配对阴性对照
  变异特异增量 = AUC(Δ) - AUC(Δ_pseudo)

【跨家族的恒等式对照（本设计最硬的一条）】
  蛋白语言模型的输入是蛋白序列。同义变异不改变蛋白 ⇒ ref 与 alt 的蛋白序列
  **逐位相同** ⇒ 嵌入必然相同 ⇒ Δ ≡ 0 ⇒ AUC(Δ) ≡ 0.5。
  这不是估计，是恒等式。凡 pLM 的 AUC(Δ) 偏离 0.5，说明管道有 bug。
  核酸模型（cLM / DNA LM / 字符级 CDS LM）没有这个约束，Δ 可以非零。
  ⇒ pLM 家族构成"阴性对照家族"，核酸模型构成"可检出家族"，
     两个家族的对比本身就是结论。

【自检（每个模型都跑，失败即判该模型不可用）】
  1. UNK 计数：分词策略自动搜索，选 UNK 最低者
  2. 阳性（核酸模型）：mean|ref-alt| 必须 > 1e-6，否则模型未感知变异
  3. 阴性（蛋白模型）：mean|ref-alt| 必须 < 1e-5，否则序列构造有错

【划分】random（分层 5 折）/ LOGO（按基因分组 5 折）
【CI】按基因聚类 bootstrap 1000 次
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)
# 服务器无外网：必须离线，否则 transformers 会对每个模型死等 5 次重试
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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

# ---------- 载入 E1 严格版已验证的序列（链向已定、框已定） ----------
_usable = np.load(OUT / "e1s_usable.npy").astype(bool)
ref_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_ref.npy", allow_pickle=True)) if _usable[i]]
alt_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_alt.npy", allow_pickle=True)) if _usable[i]]
pse_seqs = [s for i, s in enumerate(np.load(OUT / "e1s_pseudo.npy", allow_pickle=True)) if _usable[i]]
_vix = np.load(OUT / "e1s_variant_index.npy")[_usable]
_phase = np.load(OUT / "e1s_phase_pick.npy")[_usable]
labels = np.load(OUT / "e1s_labels.npy")[_usable].astype(int)
genes = np.array([g for i, g in enumerate(np.load(OUT / "e1s_genes.npy", allow_pickle=True)) if _usable[i]],
                 dtype=object)
n = len(labels)
print("可用样本 %d | 标签 %s | 基因 %d" % (n, np.bincount(labels), len(set(genes))), flush=True)

# ---------- 蛋白翻译（pLM 用） ----------
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


prot_ref = [to_prot(s, _vix[i], _phase[i]) for i, s in enumerate(ref_seqs)]
prot_alt = [to_prot(s, _vix[i], _phase[i]) for i, s in enumerate(alt_seqs)]
prot_pse = [to_prot(s, _vix[i], _phase[i]) for i, s in enumerate(pse_seqs)]
_same = sum(1 for a, b in zip(prot_ref, prot_alt) if a == b)
print("[构造自检] ref 蛋白 == alt 蛋白: %d / %d" % (_same, n))
assert _same == n, "同义变异的蛋白必须相同"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("设备:", device, flush=True)

# ---------- 分词策略（自动搜索，选 UNK 最低者） ----------
def _codons(x):
    return " ".join(x[i:i + 3] for i in range(0, len(x) - len(x) % 3, 3))


STRATEGIES = {
    "raw": lambda s: s,
    "upper": lambda s: s.upper(),
    "rna": lambda s: s.upper().replace("T", "U"),
    "rna_codon": lambda s: _codons(s.upper().replace("T", "U")),
    "dna_codon": lambda s: _codons(s.upper()),
}


def pick_strategy(tok, seqs, sample=24):
    """用小批样本试各分词策略，返回 UNK 最低者"""
    if tok.unk_token_id is None:
        return "raw", 0
    sub = seqs[:sample]
    best, best_unk = "raw", None
    for name, fn in STRATEGIES.items():
        try:
            tot = 0
            for s in sub:
                ids = tok(fn(s))["input_ids"]
                tot += sum(1 for t in ids if t == tok.unk_token_id)
            if best_unk is None or tot < best_unk:
                best, best_unk = name, tot
        except Exception:
            continue
    return best, int(best_unk or 0)


# ---------- 模型面板（V1 面板中本地可加载者） ----------
# kind: clm=密码子/核酸语言模型（可见变异）  plm=蛋白语言模型（恒等式约束）
MODELS = [
    # 密码子级 cLM —— 论文的主角
    {"id": "codonbert",        "path": "lhallee/CodonBERT",                    "kind": "clm", "bs": 32},
    {"id": "encodon-80m",      "path": "goodarzilab/encodon-80m",               "kind": "clm", "bs": 32},
    {"id": "encodon-620m",     "path": "goodarzilab/encodon-620m",              "kind": "clm", "bs": 16},
    {"id": "codontransformer", "path": "adibvafa/CodonTransformer",             "kind": "clm", "bs": 32},
    {"id": "mrnabert",         "path": "YYLY66/mRNABERT",                       "kind": "clm", "bs": 32},
    {"id": "mistral-codon-117m", "path": "RaphaelMourad/Mistral-Codon-v1-117M", "kind": "clm", "bs": 32},
    {"id": "mistral-codon-16m",  "path": "RaphaelMourad/Mistral-Codon-v1-16M",  "kind": "clm", "bs": 32},
    {"id": "decodon-200m",     "path": "goodarzilab/decodon-200M",              "kind": "clm", "bs": 32},
    # 字符级 CDS LM —— 也读核酸，应可见变异
    {"id": "cdsbert",          "path": "GleghornLab/cdsBERT",                   "kind": "clm", "bs": 32},
    {"id": "cdsbert-plus",     "path": "GleghornLab/cdsBERT-plus",              "kind": "clm", "bs": 32},
    # DNA LM —— 读核酸，但非密码子分词
    {"id": "nt-v2-500m",       "path": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
     "kind": "clm", "bs": 16},
    {"id": "nt-v2-50m",        "path": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
     "kind": "clm", "bs": 32},
    # 蛋白 LM —— 恒等式阴性对照家族
    {"id": "esm1b-650m",       "path": "facebook/esm1b_t33_650M_UR50S",         "kind": "plm", "bs": 16},
    {"id": "esm2-650m",        "path": "facebook/esm2_t33_650M_UR50D",          "kind": "plm", "bs": 16},
    {"id": "esm2-8m",          "path": "facebook/esm2_t6_8M_UR50D",             "kind": "plm", "bs": 64},
    {"id": "prot_t5",          "path": "Rostlab/prot_t5_xl_half_uniref50-enc",  "kind": "plm", "bs": 8},
]


@torch.no_grad()
def embed(tok, mdl, seqs, kind, strategy, bs=32):
    fn = STRATEGIES[strategy]
    out = []
    for i in range(0, len(seqs), bs):
        batch = [fn(s) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        if kind == "plm":
            m[:, 0] = 0.0                       # 去掉 CLS
            idx = enc["attention_mask"].sum(1) - 1
            for r, j in enumerate(idx):
                m[r, int(j)] = 0.0              # 去掉 EOS
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0)


# ---------- 评估 ----------
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
    if split == "random":
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    else:
        folds = list(GroupKFold(5).split(X, y, groups))
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
            oof = oof_scores(X, y, groups, split)
            a, lo, hi = auc_ci(y, oof, groups)
            res["%s_%s" % (arm, split)] = {
                "auc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return res


def paired_delta(y, groups, Xa, Xb, split, n_boot=1000, seed=11):
    """Δ = AUC(A) - AUC(B)，按基因聚类 bootstrap"""
    oa = oof_scores(Xa, y, groups, split)
    ob = oof_scores(Xb, y, groups, split)
    da = roc_auc_score(y, oa)
    db = roc_auc_score(y, ob)
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


all_res = {}
for mi in MODELS:
    mid = mi["id"]
    print("\n========== %s (%s) ==========" % (mid, mi["kind"]), flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(mi["path"], trust_remote_code=True)
    except Exception as e:
        print("  tokenizer 加载失败，跳过: %s" % str(e)[:160], flush=True)
        all_res[mid] = {"status": "tokenizer_load_failed", "error": str(e)[:300]}
        continue
    try:
        try:
            mdl = AutoModel.from_pretrained(mi["path"], trust_remote_code=True)
        except Exception:
            from transformers import AutoModelForMaskedLM
            mdl = AutoModelForMaskedLM.from_pretrained(mi["path"], trust_remote_code=True)
        mdl = mdl.to(device).eval()
    except Exception as e:
        print("  模型加载失败，跳过: %s" % str(e)[:160], flush=True)
        all_res[mid] = {"status": "model_load_failed", "error": str(e)[:300]}
        continue

    try:
        seqs3 = [prot_ref, prot_alt, prot_pse] if mi["kind"] == "plm" else [ref_seqs, alt_seqs, pse_seqs]
        strategy, unk = pick_strategy(tok, seqs3[0])
        print("  分词策略: %s | 样例 UNK: %d" % (strategy, unk), flush=True)

        E_ref = embed(tok, mdl, seqs3[0], mi["kind"], strategy, mi.get("bs", 32))
        E_alt = embed(tok, mdl, seqs3[1], mi["kind"], strategy, mi.get("bs", 32))
        E_pse = embed(tok, mdl, seqs3[2], mi["kind"], strategy, mi.get("bs", 32))
        d = float(np.abs(E_ref - E_alt).mean())

        ok = True
        if mi["kind"] == "clm":
            ok = d > 1e-6
            print("  [自检 阳性] 核酸模型要求 mean|ref-alt| > 1e-6 ⇒ %.3e %s"
                  % (d, "通过" if ok else "★失败：模型未感知变异"), flush=True)
        else:
            ok = d < 1e-5
            print("  [自检 阴性] 蛋白模型要求 mean|ref-alt| < 1e-5（恒等式）⇒ %.3e %s"
                  % (d, "通过（Δ≡0，无变异特异分量）" if ok else "★失败：序列构造有错"), flush=True)
        if not ok:
            all_res[mid] = {"status": "selfcheck_failed", "mean_abs_ref_alt": d,
                            "strategy": strategy, "unk": unk}
            del mdl
            gc.collect()
            torch.cuda.empty_cache()
            continue

        np.save(OUT / ("e1s_%s_ref.npy" % mid), E_ref)
        np.save(OUT / ("e1s_%s_alt.npy" % mid), E_alt)
        np.save(OUT / ("e1s_%s_pse.npy" % mid), E_pse)

        Xdict = {
            "ref": E_ref,
            "alt": E_alt,
            "diff": E_alt - E_ref,
            "pair": np.concatenate([E_ref, E_alt], 1),
            "pseudo_diff": E_pse - E_ref,
        }
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
        r["_selfcheck"] = {"mean_abs_ref_alt": d, "unk": int(unk),
                           "strategy": strategy, "kind": mi["kind"], "status": "ok"}
        all_res[mid] = r

        with open(OUT / "e1_multimodel_results.json", "w") as f:
            json.dump(all_res, f, indent=1, ensure_ascii=False)
        print("  [已落盘]", flush=True)
    except Exception as e:
        print("  运行失败: %s" % str(e)[:300], flush=True)
        all_res[mid] = {"status": "run_failed", "error": str(e)[:300]}

    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT / "e1_multimodel_results.json", "w") as f:
    json.dump(all_res, f, indent=1, ensure_ascii=False)
print("\n全部完成:", OUT / "e1_multimodel_results.json")
