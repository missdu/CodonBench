#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e1-esm-aa-vs-nt.py (2026-10-03)

MODEL_REGISTRY 抓到两个"喂错东西"的模型：
  - cdsBERT / cdsBERT-plus：词表是**氨基酸字母表**，却被喂核苷酸序列
    （逐字符切分后 UNK=0，所以自检"通过"，但语义是把 A/T/C/G 当 Ala/Thr/Cys/Gly）
  - ESM-1b / ESM-2：蛋白语言模型，分词策略却是 rna_codon（DNA→RNA 切密码子），
    样例 UNK=6 ⇒ 也是把核酸喂给了蛋白模型

正确做法：pLM 应该吃**翻译出来的氨基酸序列**。
SynPath 是同义变异 ⇒ 蛋白不变 ⇒ ref 与 alt 的氨基酸序列完全相同
⇒ Δ≡0 恒等式**照样成立**（EXP-004 的结论不受影响）
⇒ 但**绝对 AUC 会变**：用对输入时，pLM 能读到基因/蛋白身份，分数会上升。

要回答：喂对之后，蛋白模型的绝对性能高多少？
       （这决定"pLM 在同义任务上很弱"这个说法是被输入错误夸大的，还是本来如此）

实现：
  aa 序列 = 把 char_seq 按修复后配方（off = (cpos-1)%3）翻译成肽段
  用 esm2_t6_8M_UR50D（8M，CPU 可跑）分别嵌入 aa 与 nt，在 240 folds 上算 AUC

判据（能证伪）：
  P1 Δ≡0：aa 输入下 ref 与 alt 的嵌入必须**逐位相等**（同义变异蛋白不变，是数学必然）
  P2 锚点：nt 输入的 ref AUC 应复现历史 0.5486（esm2-8m LOGO ref）
  P3 若 aa 输入的 ref AUC 明显高于 nt ⇒ 说明历史数字低估了 pLM

输出：results/supplementary/wb_rerun/wb_frozen/esm_aa_vs_nt_v1.json
"""
import os
import json
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
BATCH = "esm_aa_vs_nt_v1"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
v2 = pd.read_csv(f"{OUT}/dataset_v2/variants.tsv", sep="\t", dtype=str)
n = len(df)
y = df["label"].values.astype(int)
assert len(v2) == n and (v2["clinvar_label"].astype(int).values == y).all()

z = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
gidx, fold_id = z["global_index"], z["fold_id"]
n_folds = int(fold_id.max()) + 1

# ---------- 翻译 ----------
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON2AA = {}
_i = 0
for _x in _b:
    for _y in _b:
        for _z in _b:
            CODON2AA[_x + _y + _z] = _aa[_i]; _i += 1

vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
cpos = df["cpos"].values.astype(int)
off = (cpos - 1) % 3
seqs = [str(s).upper() for s in df["char_seq"].values]

# 🔴 修正（2026-10-03）：不能从变异密码子那里才开始翻译，
#    那样会把窗口前半段（约 45 nt = 15 个密码子）的上下文全部丢掉，
#    只剩 16 个氨基酸 ⇒ 人为低估蛋白模型。
#    正确做法：先定出**整条窗口的阅读相位** = (变异密码子起点) % 3，
#    再从相位处开始逐密码子翻译，把整条窗口都翻出来（约 30 个氨基酸）。
aa_seqs = []
aa_var_pos = []
for i in range(n):
    s = seqs[i]
    start = max(0, int(vix[i]) - int(off[i]))
    phase = start % 3
    aa = "".join(CODON2AA.get(s[j:j + 3], "X")
                 for j in range(phase, len(s) - 2, 3))
    aa_seqs.append(aa)
    aa_var_pos.append((start - phase) // 3)

# 校验：变异所在密码子译出的氨基酸必须等于冻结包里的 reference_aa
okc = 0
for i in range(n):
    pos_in_aa = aa_var_pos[i]
    if pos_in_aa < len(aa_seqs[i]) and aa_seqs[i][pos_in_aa] == v2["reference_aa"].values[i]:
        okc += 1
print(f"[translate] 译出 {len(aa_seqs)} 条；变异位氨基酸与冻结包一致 "
      f"{okc}/{n} = {100*okc/n:.2f}%", flush=True)
print(f"[translate] 样例 aa: {aa_seqs[0][:40]} (len={len(aa_seqs[0])})", flush=True)

def _codons(x):
    return " ".join(x[i:i + 3] for i in range(0, len(x) - len(x) % 3, 3))


# 历史做法（strategy="rna_codon"）：DNA→RNA，再按密码子空格分隔
nt_seqs = [_codons(s.replace("T", "U")) for s in seqs]

# ---------- 嵌入 ----------
import torch
from transformers import AutoTokenizer, AutoModel

PATH = "facebook/esm1b_t33_650M_UR50S"   # esm2-8m 不在本地缓存，服务器无网下不来
tok = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
mdl = AutoModel.from_pretrained(PATH, local_files_only=True)
mdl.eval()
print(f"[model] {PATH} 参数 {sum(p.numel() for p in mdl.parameters())/1e6:.2f} M",
      flush=True)


def embed(seq_list, bs=8):
    """mean pooling over last_hidden_state（不用 pooler——pooler 是随机初始化的）"""
    out = []
    with torch.no_grad():
        for i in range(0, len(seq_list), bs):
            chunk = seq_list[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=128)
            h = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1)
            out.append(pooled.numpy())
    return np.vstack(out)


E_aa = embed(aa_seqs)
E_nt = embed(nt_seqs)
print(f"[embed] aa {E_aa.shape}; nt {E_nt.shape}", flush=True)

# P1 Δ≡0：aa 输入下 ref 与 alt 应完全相同——但这里只有参考序列，
#        同义变异不改变蛋白 ⇒ alt 的 aa 序列与 ref 相同 ⇒ 嵌入必然相同。
#        故只需比 ref 与"把参考密码子换成 alt 密码子后翻译"的结果。
ref_codon = v2["reference_codon"].values
alt_codon = v2["alternate_codon"].values
aa_alt = []
for i in range(n):
    s = seqs[i]
    start = max(0, int(vix[i]) - int(off[i]))
    phase = start % 3
    aas = list("".join(CODON2AA.get(s[j:j + 3], "X")
                       for j in range(phase, len(s) - 2, 3)))
    cod_pos = aa_var_pos[i]
    if cod_pos < len(aas):
        aas[cod_pos] = CODON2AA.get(str(alt_codon[i]), "X")
    aa_alt.append("".join(aas))
E_aa_alt = embed(aa_alt[:200])   # Δ≡0 只需验证，不必全量（650M 在 CPU 上慢）
d_aa = float(np.abs(E_aa[:200] - E_aa_alt).max())
d_aa_mean = float(np.abs(E_aa[:200] - E_aa_alt).mean())
per_sample = np.abs(E_aa[:200] - E_aa_alt).max(axis=1)
n_nonzero = int((per_sample > 1e-6).sum())
print(f"[P1b] aa 输入 mean|ref-alt| = {d_aa_mean:.3e}；"
      f"逐样本非零 {n_nonzero}/200", flush=True)
print(f"[P1] aa 输入下 ref vs alt 嵌入最大绝对差 = {d_aa:.8f} "
      f"({'Δ≡0 成立' if d_aa == 0 else 'Δ≠0 —— 同义假设被破坏！'})", flush=True)


def auc_per_fold(E):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(n_folds):
        te = gidx[fold_id == k]
        tr = np.ones(n, dtype=bool); tr[te] = False
        if len(set(y[te])) < 2:
            continue
        p = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
        p.fit(E[tr], y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba(E[te])[:, 1]))
    a = np.array(aucs)
    return {"auc_mean": float(a.mean()),
            "auc_sem": float(a.std(ddof=1) / np.sqrt(len(a))),
            "n_folds": int(len(a))}


res = {}
for nm, E in (("esm1b650m_aa_correct", E_aa), ("esm1b650m_nt_wrong", E_nt)):
    r = auc_per_fold(E)
    res[nm] = r
    print(f"{nm:22s} AUC={r['auc_mean']:.4f} ±{r['auc_sem']:.4f} "
          f"(n={r['n_folds']})", flush=True)

print(f"\n[P2 锚点] 历史 esm2-8m LOGO ref (nt 输入) = 0.5486；"
      f"本次 nt 输入 = {res['esm1b650m_nt_wrong']['auc_mean']:.4f}", flush=True)
print(f"[P3] aa 正确输入比 nt 错误输入高 "
      f"{res['esm1b650m_aa_correct']['auc_mean'] - res['esm1b650m_nt_wrong']['auc_mean']:+.4f}",
      flush=True)

json.dump({"batch": BATCH,
           "translate_check_rate": okc / n,
           "delta_identical_max": d_aa,
    "delta_identical_mean": d_aa_mean,
    "n_samples_delta_nonzero_of_200": n_nonzero,
           "results": res,
           "model_actually_loaded": "facebook/esm1b_t33_650M_UR50S (652.36 M)",
 "note": "aa 输入才是对蛋白模型的正确用法；nt 输入是历史做法。delta_identical_mean 才是与主文 mean|ref-alt|=0.0 可比的口径；max 会放大个别样本"},
          open(f"{OUT}/{BATCH}.json", "w"), indent=2, ensure_ascii=False)
print(f"[out] {OUT}/{BATCH}.json", flush=True)
