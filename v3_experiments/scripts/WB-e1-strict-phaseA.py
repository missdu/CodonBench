# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— Phase A：数据自检（纯 CPU，不加载模型）

要回答的问题：
  Q-A 这 2840 条是不是真的"同义"？
      判据 1：Name 字段的 p. 记号是否以 '=' 结尾（如 p.Lys599=）
      判据 2：用 cpos 推密码子框，翻译 ref 窗口与 alt 窗口，蛋白串是否逐条相同
  Q-B 变异位置映射对不对？
      判据：char_seq[45] == ReferenceAlleleVCF 的比例
      判据：codon_seq[(cpos-1)%3] == ReferenceAlleleVCF 的比例（验证 cpos 即编码位置）
  Q-C 有多少条 ref 与 alt 窗口其实相同（无效样本）？

这条自检的意义（论文里是恒等式级别的论证）：
  同义变异 ⇒ 编码蛋白不变 ⇒ **任何蛋白语言模型在 ref 与 alt 上的输入完全相同**
  ⇒ pLM 对该任务的贡献必然 100% 来自序列背景（基因身份），变异特异分量为零。
  这不是估计，是构造。cLM 则不受此约束（核苷酸变了），所以 E1 要测的是：
  cLM 到底有没有把"变了"这件事变成可迁移的预测力。

只读，不写任何数据文件（只打印 + 存一份 json 报告到 wb_rerun/）。
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

BASE = Path(os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp"))
SYNPATH = BASE / "results" / "unified_eval" / "synpath_data"
OUT = BASE / "results" / "supplementary" / "wb_rerun"
OUT.mkdir(exist_ok=True)

df = pd.read_parquet(SYNPATH / "synpath_variants.parquet")
n = len(df)
print("SynPath 样本数: %d" % n)

CODON_TABLE = {}
_bases = "TCAG"
_aas = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
_i = 0
for b1 in _bases:
    for b2 in _bases:
        for b3 in _bases:
            CODON_TABLE[b1 + b2 + b3] = _aas[_i]
            _i += 1


def translate(s):
    return "".join(CODON_TABLE.get(s[i:i + 3], "X")
                   for i in range(0, len(s) - len(s) % 3, 3))


rep = {}

# ---------- Q-A1: p. 记号 ----------
name = df["Name"].astype(str)
p_eq = name.str.contains(r"p\.[A-Z][a-z]{2}\d+=", regex=True)
print("\n[Q-A1] Name 含 p.Xxx###= 记号: %d / %d (%.2f%%)"
      % (p_eq.sum(), n, 100 * p_eq.mean()))
rep["p_synonymous_notation"] = {"n": int(p_eq.sum()), "total": n}

# 看看不匹配的样本长什么样
bad = name[~p_eq].head(8).tolist()
print("  不匹配样例:", bad)
rep["p_notation_counterexamples"] = bad

# 顺带看 p. 记号里有没有非 '='（即错义/无义被误纳入）
p_missense = name.str.contains(r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", regex=True)
print("  含 p.Xxx###Yyy（非同义写法）: %d" % int(p_missense.sum()))
rep["p_missense_notation"] = int(p_missense.sum())

# ---------- Q-B: 位置映射 ----------
char_seq = df["char_seq"].astype(str).values
ref_base = df["ReferenceAlleleVCF"].astype(str).values
alt_base = df["AlternateAlleleVCF"].astype(str).values
codon_seq = df["codon_seq"].astype(str).values
cpos = df["cpos"].values

CENTER = 45
lens = np.array([len(s) for s in char_seq])
print("\n[Q-B] char_seq 长度: min %d / max %d / 众数 %d"
      % (lens.min(), lens.max(), pd.Series(lens).mode()[0]))
rep["char_seq_len"] = {"min": int(lens.min()), "max": int(lens.max())}

COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def comp(b):
    return COMP.get(b, "N")


ok_center = np.array([
    len(s) > CENTER and s[CENTER] == r
    for s, r in zip(char_seq, ref_base)])
ok_center_comp = np.array([
    len(s) > CENTER and s[CENTER] == comp(r)
    for s, r in zip(char_seq, ref_base)])
print("  char_seq[45] == ReferenceAllele      : {} / {} ({:.2f}%)".format(
    ok_center.sum(), n, 100 * ok_center.mean()))
print("  char_seq[45] == complement(Reference): {} / {} ({:.2f}%)".format(
    ok_center_comp.sum(), n, 100 * ok_center_comp.mean()))
print("    两者之一成立: {} / {} ({:.2f}%)".format(
    int((ok_center | ok_center_comp).sum()), n,
    100 * (ok_center | ok_center_comp).mean()))
print("    ⇒ 若第二列显著 >0，说明负链基因的窗口取了反向互补，")
print("      构造 alt 序列时必须先定链，否则一半样本的'变异'是错的。")
rep["center_base_match"] = {"n": int(ok_center.sum()), "total": n}
rep["center_base_match_complement"] = {"n": int(ok_center_comp.sum()), "total": n}
rep["center_base_match_either"] = {
    "n": int((ok_center | ok_center_comp).sum()), "total": n}

off = ((cpos - 1) % 3).astype(int)
ok_codon = np.array([
    len(c) == 3 and 0 <= o < 3 and c[o] == r
    for c, o, r in zip(codon_seq, off, ref_base)])
ok_codon_comp = np.array([
    len(c) == 3 and 0 <= o < 3 and c[o] == comp(r)
    for c, o, r in zip(codon_seq, off, ref_base)])
print("  codon_seq[(cpos-1)%3] == ReferenceAllele      : {} / {} ({:.2f}%)".format(
    ok_codon.sum(), n, 100 * ok_codon.mean()))
print("  codon_seq[(cpos-1)%3] == complement(Reference): {} / {} ({:.2f}%)".format(
    ok_codon_comp.sum(), n, 100 * ok_codon_comp.mean()))
print("    ⇒ 该比例高即说明 cpos 是编码区坐标、codon_seq 是变异所在密码子；")
print("      同时也指示 codon_seq 是正链写法还是已取反向互补。")
rep["codon_frame_match"] = {"n": int(ok_codon.sum()), "total": n}
rep["codon_frame_match_complement"] = {"n": int(ok_codon_comp.sum()), "total": n}

# ---------- Q-A2: 翻译等价性（核心） ----------
# 先定链：正链用原碱基，负链用互补碱基；无法判定的样本单独计数
strand = np.zeros(n, dtype=int)   # +1 正链, -1 负链, 0 未定
for i, (s, r) in enumerate(zip(char_seq, ref_base)):
    if len(s) <= CENTER:
        continue
    if s[CENTER] == r:
        strand[i] = 1
    elif s[CENTER] == comp(r):
        strand[i] = -1
print("\n[链向判定] 正链 %d | 负链 %d | 未定 %d"
      % ((strand == 1).sum(), (strand == -1).sum(), (strand == 0).sum()))
rep["strand"] = {"plus": int((strand == 1).sum()),
                 "minus": int((strand == -1).sum()),
                 "unknown": int((strand == 0).sum())}

# 构造 alt 窗口：把 char_seq[CENTER] 换成（按链取过互补的）alt 碱基
alt_seq = []
for s, a, st in zip(char_seq, alt_base, strand):
    if len(s) <= CENTER or st == 0:
        alt_seq.append(s)
        continue
    b = a if st == 1 else comp(a)
    alt_seq.append(s[:CENTER] + b + s[CENTER + 1:])
alt_seq = np.array(alt_seq)

same_window = np.array([a == b for a, b in zip(char_seq, alt_seq)])
print("\n[Q-C] ref 窗口 == alt 窗口（无效样本）: %d / %d" % (int(same_window.sum()), n))
rep["window_identical"] = int(same_window.sum())

def revcomp(s):
    return "".join(comp(c) for c in reversed(s))


# 翻译必须在**编码链**上做：负链样本的 char_seq 是反向互补，
# 直接按自身读框翻译会把"同义"变成"不同义"（例 AAA→AAG 的反向互补是 TTT→CTT）。
coding_ref, coding_alt, coding_center = [], [], []
for s, a, st in zip(char_seq, alt_seq, strand):
    if st == -1:
        coding_ref.append(revcomp(s))
        coding_alt.append(revcomp(a))
        coding_center.append(len(s) - 1 - CENTER)
    else:
        coding_ref.append(s)
        coding_alt.append(a)
        coding_center.append(CENTER)

# 按推定的阅读框翻译：变异位点在 coding_center，其密码子内偏移 = off
prot_ref, prot_alt, frame_ok = [], [], 0
for s, a, o, cix in zip(coding_ref, coding_alt, off, coding_center):
    start = cix - int(o)
    if start < 0 or len(s) - start < 3:
        prot_ref.append("")
        prot_alt.append("")
        continue
    frame_ok += 1
    sr = s[start:]
    sa = a[start:]
    L = min(len(sr), len(sa))
    L -= L % 3
    prot_ref.append(translate(sr[:L]))
    prot_alt.append(translate(sa[:L]))
prot_ref = np.array(prot_ref)
prot_alt = np.array(prot_alt)
eq = prot_ref == prot_alt
print("\n[Q-A2 核心] 推定阅读框可翻译样本: %d / %d" % (frame_ok, n))
print("  翻译后 ref 蛋白 == alt 蛋白: %d / %d (%.2f%%)"
      % (eq.sum(), frame_ok, 100 * eq.mean() / max(frame_ok, 1)))
rep["translation_equal"] = {"n": int(eq.sum()), "total": int(frame_ok)}

# 不等的样例，看看是错义还是框推定错
bad_idx = np.where(~eq)[0][:5]
for i in bad_idx:
    print("  ✗ idx %d | %s | cpos %s off %d | codon %s | ref %s alt %s"
          % (i, df["Name"].iloc[i][:40], cpos[i], off[i], codon_seq[i],
             ref_base[i], alt_base[i]))
    print("      prot_ref[:20] %s" % prot_ref[i][:20])
    print("      prot_alt[:20] %s" % prot_alt[i][:20])

# 用 codon_seq 直接判同义（不依赖窗口框）
def is_syn_by_table(c, r, a, o, st):
    if len(c) != 3 or not (0 <= o < 3) or st == 0:
        return None
    r2 = r if st == 1 else comp(r)
    a2 = a if st == 1 else comp(a)
    if c[o] != r2:
        return None
    mut = c[:o] + a2 + c[o + 1:]
    return CODON_TABLE.get(c, "X") == CODON_TABLE.get(mut, "X")


syn_direct = [is_syn_by_table(c, r, a, o, st)
              for c, r, a, o, st in zip(codon_seq, ref_base, alt_base, off, strand)]
valid = [x for x in syn_direct if x is not None]
print("\n[Q-A3] 用密码子表直接判同义（codon_seq + 参考/变异碱基）:")
print("  可判定样本 %d / %d | 其中同义 %d (%.2f%%)"
      % (len(valid), n, sum(valid), 100 * sum(valid) / max(len(valid), 1)))
rep["codon_table_synonymous"] = {"syn": int(sum(valid)), "judgeable": len(valid)}

# 标签与基因
print("\n[数据概况]")
print("  标签: %s" % df["label"].value_counts().to_dict())
print("  基因数: %d | 转录本数: %d"
      % (df["gene_symbol"].nunique(), df["tx_id"].nunique()))
rep["n_genes"] = int(df["gene_symbol"].nunique())
rep["n_tx"] = int(df["tx_id"].nunique())
rep["labels"] = {str(k): int(v) for k, v in df["label"].value_counts().items()}

with open(OUT / "e1_strict_phaseA_report.json", "w") as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print("\n报告已存:", OUT / "e1_strict_phaseA_report.json")
