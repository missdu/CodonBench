# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— Phase B：提嵌入 + 跑臂（GPU）

依赖 Phase A2 产出的 e1s_*.npy（编码链窗口，已定链、已定框）

【本实验要回答的问题】
  同义变异的可预测性里，有多少来自"序列背景"（这个变异长在哪个基因里），
  有多少来自"变异本身"（这个碱基替换带来的变化）？

【臂】
  ref     —— 只给参考序列嵌入（完全不含变异信息）⇒ 背景可预测性
  alt     —— 给变异序列嵌入
  diff    —— alt − ref（纯粹的"变异引入的变化"）
  pair    —— [ref, alt] 拼接
  pseudo  —— pseudo − ref（假变异：同窗口内偏移 15 nt 处换一个碱基）⇒ 阴性对照

【两条内建的管道自检】（这是"严格版"区别于原版的地方）
  自检 1（阳性）：cLM 的 ref 与 alt 嵌入必须不同（mean|差| > 0），
                  否则说明分词错（如 CodonBERT 的 RNA 密码子词表遇 DNA 全 UNK）
  自检 2（阴性）：pLM 的 ref 与 alt **蛋白序列相同**（同义变异的定义）
                  ⇒ 其嵌入必须逐位相等（mean|差| ≈ 0）。
                  若不等 ⇒ 序列构造或框推定有错，整个实验作废。

【划分】
  random —— 5 折分层随机 CV，池化 out-of-fold
  LOGO   —— 5 折按基因分组 CV，池化 out-of-fold
  CI     —— 按基因聚类 bootstrap，1000 次
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
OUT = BASE / "results" / "supplementary" / "wb_rerun"

ref_seqs = list(np.load(OUT / "e1s_ref.npy", allow_pickle=True))
alt_seqs = list(np.load(OUT / "e1s_alt.npy", allow_pickle=True))
pse_seqs = list(np.load(OUT / "e1s_pseudo.npy", allow_pickle=True))
strand = np.load(OUT / "e1s_strand.npy")
vix = np.load(OUT / "e1s_variant_index.npy")
phase = np.load(OUT / "e1s_phase_pick.npy")
usable = np.load(OUT / "e1s_usable.npy")
labels = np.load(OUT / "e1s_labels.npy")
genes = list(np.load(OUT / "e1s_genes.npy", allow_pickle=True))

ref_seqs = [ref_seqs[i] for i in range(len(ref_seqs)) if usable[i]]
alt_seqs = [alt_seqs[i] for i in range(len(alt_seqs)) if usable[i]]
pse_seqs = [pse_seqs[i] for i in range(len(pse_seqs)) if usable[i]]
vix = vix[usable]
phase = phase[usable]
labels = labels[usable].astype(int)
genes = np.array([genes[i] for i in range(len(usable)) if usable[i]], dtype=object)
n = len(labels)
print("可用样本 %d | 标签 %s | 基因 %d" % (n, np.bincount(labels), len(set(genes))))

# ---------- 蛋白翻译（给 pLM 用） ----------
_bases = "TCAG"
_aas = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODON = {}
_i = 0
for b1 in _bases:
    for b2 in _bases:
        for b3 in _bases:
            CODON[b1 + b2 + b3] = _aas[_i]
            _i += 1


def to_prot(s, ix, ph):
    st = ix - ph
    if st < 0:
        st = 0
    t = s[st:]
    L = len(t) - len(t) % 3
    return "".join(CODON.get(t[i:i + 3], "X") for i in range(0, L, 3))


prot_ref = [to_prot(s, vix[i], phase[i]) for i, s in enumerate(ref_seqs)]
prot_alt = [to_prot(s, vix[i], phase[i]) for i, s in enumerate(alt_seqs)]
prot_pse = [to_prot(s, vix[i], phase[i]) for i, s in enumerate(pse_seqs)]
same = sum(1 for a, b in zip(prot_ref, prot_alt) if a == b)
print("[构造自检] ref 蛋白 == alt 蛋白: %d / %d" % (same, n))
assert same == n, "同义变异的蛋白必须相同，否则框推定有错"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("设备:", device)


def to_rna_codons(x):
    r = x.replace("T", "U")
    return " ".join([r[i:i + 3] for i in range(0, len(r) - len(r) % 3, 3)])


MODELS = [
    {"id": "codonbert", "path": "lhallee/CodonBERT", "kind": "clm"},
    {"id": "esm1b-650m", "path": "facebook/esm1b_t33_650M_UR50S", "kind": "plm"},
]


def load_model(mi):
    tok = AutoTokenizer.from_pretrained(mi["path"], trust_remote_code=True)
    try:
        mdl = AutoModel.from_pretrained(mi["path"], trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForMaskedLM
        mdl = AutoModelForMaskedLM.from_pretrained(mi["path"], trust_remote_code=True)
    mdl = mdl.to(device).eval()
    return tok, mdl


@torch.no_grad()
def embed(tok, mdl, seqs, kind, bs=32):
    out = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        if kind == "clm":
            batch = [to_rna_codons(s) for s in batch]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        if kind == "plm":
            # 去掉 CLS / EOS
            m[:, 0] = 0.0
            idx = enc["attention_mask"].sum(1) - 1
            for r, j in enumerate(idx):
                m[r, int(j)] = 0.0
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, 0)


# ---------- 评估 ----------
def auc_ci(y, s, groups, n_boot=1000, seed=7):
    """按基因聚类 bootstrap 的 AUC 与 95% CI"""
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


def run_arms(Xdict, y, groups, tag):
    """对每个特征矩阵跑 random / LOGO 两种划分"""
    res = {}
    for split_name, splitter in [
        ("random", StratifiedKFold(5, shuffle=True, random_state=42)),
        ("LOGO", GroupKFold(5)),
    ]:
        for arm, X in Xdict.items():
            X = np.asarray(X, dtype=np.float64)
            oof = np.zeros(len(y))
            if split_name == "random":
                folds = list(splitter.split(X, y))
            else:
                folds = list(splitter.split(X, y, groups))
            for tr, te in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, C=1.0))
                clf.fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            a, lo, hi = auc_ci(y, oof, groups)
            res["%s_%s" % (arm, split_name)] = {
                "auc": round(a, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    print("\n--- %s ---" % tag)
    for k in sorted(res):
        v = res[k]
        print("  %-22s AUC %.4f  [%.4f, %.4f]" % (k, v["auc"], v["ci_lo"], v["ci_hi"]))
    return res


all_res = {}
for mi in MODELS:
    mid = mi["id"]
    print("\n========== %s (%s) ==========" % (mid, mi["kind"]))
    try:
        tok, mdl = load_model(mi)
    except Exception as e:
        print("  加载失败，跳过: %s" % str(e)[:200])
        continue

    if mi["kind"] == "clm":
        seqs3 = [ref_seqs, alt_seqs, pse_seqs]
    else:
        seqs3 = [prot_ref, prot_alt, prot_pse]

    # 分词质量自检
    s0 = seqs3[0][0]
    q = to_rna_codons(s0) if mi["kind"] == "clm" else s0
    ids = tok(q)["input_ids"]
    unk = sum(1 for x in ids if x == tok.unk_token_id)
    print("  样例 token 数 %d | UNK %d" % (len(ids), unk))

    E_ref = embed(tok, mdl, seqs3[0], mi["kind"])
    E_alt = embed(tok, mdl, seqs3[1], mi["kind"])
    E_pse = embed(tok, mdl, seqs3[2], mi["kind"])
    d = float(np.abs(E_ref - E_alt).mean())
    print("  嵌入 %s | mean|ref-alt| = %.3e" % (E_ref.shape, d))

    if mi["kind"] == "clm":
        print("  [自检1 阳性] cLM ref≠alt 要求 mean|差|>1e-6 ⇒ %s"
              % ("通过" if d > 1e-6 else "★失败：模型未感知变异"))
    else:
        print("  [自检2 阴性] pLM ref≡alt 要求 mean|差|<1e-5 ⇒ %s"
              % ("通过（恒等式成立，pLM 无变异特异分量）" if d < 1e-5
                 else "★失败：序列构造或框推定有错"))

    np.save(OUT / ("e1s_%s_ref.npy" % mid), E_ref)
    np.save(OUT / ("e1s_%s_alt.npy" % mid), E_alt)

    Xdict = {
        "ref": E_ref,
        "alt": E_alt,
        "diff": E_alt - E_ref,
        "pair": np.concatenate([E_ref, E_alt], 1),
        "pseudo_diff": E_pse - E_ref,
    }
    r = run_arms(Xdict, labels, genes, mid)
    r["_selfcheck"] = {"mean_abs_ref_alt": d, "unk": int(unk), "kind": mi["kind"]}

    # 标签置换对照：只跑 diff 臂，random 与 LOGO 各一次
    rng = np.random.RandomState(123)
    y_perm = labels.copy()
    rng.shuffle(y_perm)
    rp = run_arms({"diff": E_alt - E_ref}, y_perm, genes, mid + " [标签置换]")
    r["_permuted"] = rp

    all_res[mid] = r
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(OUT / "e1_strict_results.json", "w") as f:
    json.dump(all_res, f, indent=1, ensure_ascii=False)
print("\n结果已存:", OUT / "e1_strict_results.json")
