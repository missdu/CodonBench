# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-t1-diag3.py (2026-10-03)
验证假设：**char_seq 已经是编码链（转录本）方向，不需要再反向互补**。

证据链：
  正链基因吻合率 99.51%（不做 rc，正好对）
  负链基因吻合率  7.06%（多做了一次 rc ⇒ 拿到的是反义链密码子）
  HGVS 的 ref 碱基与 VCF 的 ReferenceAlleleVCF 只在 ~50% 样本上相同
    ⇒ VCF 列是基因组正链语义，char_seq 与 HGVS 是编码链语义

若假设成立，正确配方应为：
  cod = char_seq[vi-off : vi-off+3]           ← 不 rc
  ref_c = VCF_ref        (正链基因)
        = COMP(VCF_ref)  (负链基因)           ← 碱基取互补，不是序列取互补
判据：用 HGVS 的参考氨基酸检验，正链负链都应 ≈100%。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import re, math
import numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
n = len(df)
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
vcf_ref = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
vcf_alt = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = np.array([str(s) for s in df["char_seq"].values])
cpos = df["cpos"].values.astype(int)
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
def rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s.upper()))

PATC = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
PATP = re.compile(r"p\.([A-Za-z]{3})(\d+)(=?)([A-Za-z]{3}|Ter)?")
hg_ref = np.empty(n, dtype=object); hg_alt = np.empty(n, dtype=object)
hg_aa = np.empty(n, dtype=object); hg_syn = np.zeros(n, dtype=bool)
for i in range(n):
    m = PATC.search(str(df["Name"].iloc[i]))
    if m:
        hg_ref[i] = m.group(2); hg_alt[i] = m.group(3)
    m2 = PATP.search(str(df["Name"].iloc[i]))
    if m2:
        hg_aa[i] = m2.group(1); hg_syn[i] = (m2.group(3) == "=")

bases = "TCAG"; aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CT = {}; k = 0
for a in bases:
    for b in bases:
        for c in bases:
            CT[a + b + c] = aas[k]; k += 1
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "*"}
hg_aa1 = np.array([AA3.get(a, "?") for a in hg_aa])

# 链向：以 HGVS ref 为准（编码链语义）
plus = np.array([(hg_ref[i] == vcf_ref[i]) for i in range(n)])
minus = np.array([(hg_ref[i] == COMP.get(vcf_ref[i], "N")) for i in range(n)])
unk = ~plus & ~minus
print(f"链向（以 HGVS ref 为编码链基准）: plus={plus.sum()} minus={minus.sum()} "
      f"不可判={unk.sum()}")

print("\n=== 假设检验：不 rc 序列，碱基按链向取互补 ===")
def check(do_rc, comp_base, tag):
    ok_ref = 0; ok_syn = 0; tot = 0; syn_hg = 0
    for i in range(n):
        if hg_aa1[i] == "?":
            continue
        s = seqs[i].upper()
        if do_rc and minus[i]:
            s = rc(s)
        vi = int(vix[i]); off = int((cpos[i] - 1) % 3); st = vi - off
        if st < 0 or st + 3 > len(s):
            continue
        cod = s[st:st + 3]
        r = vcf_ref[i]; a = vcf_alt[i]
        if comp_base and minus[i]:
            r = COMP.get(r, "N"); a = COMP.get(a, "N")
        if cod[off] != r:
            continue
        acod = cod[:off] + a + cod[off + 1:]
        tot += 1
        if CT.get(cod, "X") == hg_aa1[i]:
            ok_ref += 1
        mysyn = CT.get(cod, "X") == CT.get(acod, "X")
        if hg_syn[i]:
            syn_hg += 1
            if mysyn:
                ok_syn += 1
    print(f"  {tag:38s} 参考氨基酸吻合 {ok_ref}/{tot} = {ok_ref/max(1,tot):.4f} | "
          f"同义复现 {ok_syn}/{syn_hg} = {ok_syn/max(1,syn_hg):.4f}")

check(False, False, "不rc + 碱基不变")
check(False, True,  "不rc + 负链碱基互补  ★假设")
check(True,  False, "rc   + 碱基不变（EXP-016 现行）")
check(True,  True,  "rc   + 负链碱基互补")

print("\n=== 按链向分组（★假设配方）===")
for mask, tag in ((plus, "正链"), (minus, "负链"), (unk, "不可判")):
    ok = 0; tot = 0; sy = 0; sh = 0
    for i in range(n):
        if hg_aa1[i] == "?" or not mask[i]:
            continue
        s = seqs[i].upper(); vi = int(vix[i]); off = int((cpos[i] - 1) % 3)
        st = vi - off
        if st < 0 or st + 3 > len(s):
            continue
        cod = s[st:st + 3]
        r = vcf_ref[i] if plus[i] else COMP.get(vcf_ref[i], "N")
        a = vcf_alt[i] if plus[i] else COMP.get(vcf_alt[i], "N")
        if cod[off] != r:
            continue
        acod = cod[:off] + a + cod[off + 1:]
        tot += 1
        if CT.get(cod, "X") == hg_aa1[i]:
            ok += 1
        if hg_syn[i]:
            sh += 1
            if CT.get(cod, "X") == CT.get(acod, "X"):
                sy += 1
    print(f"  {tag:8s} 参考氨基酸 {ok}/{tot} = {ok/max(1,tot):.4f} | "
          f"同义复现 {sy}/{sh} = {sy/max(1,sh):.4f}")

import json
json.dump({"plus": int(plus.sum()), "minus": int(minus.sum()), "unknown": int(unk.sum())},
          open(f"{OUT}/strand_by_hgvs.json", "w"), indent=2)
print("\n已写", f"{OUT}/strand_by_hgvs.json")
