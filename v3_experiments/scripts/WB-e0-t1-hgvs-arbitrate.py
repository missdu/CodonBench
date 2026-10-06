# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-t1-hgvs-arbitrate.py (2026-10-03)
用 **HGVS 蛋白表达**（SynPath `Name` 列，形如 p.Lys599=）作**独立第三方**，
仲裁"我从序列翻译出的氨基酸"到底对不对。

为什么需要仲裁：
  我用 (cpos-1)%3 定相位后，仍有 205 条（7.22%）翻译前后氨基酸不一致，
  但 ClinVar 的 Name 列把它们也标成同义（p.Arg33648=）。
  两边冲突，必须有一个独立判据来判谁错——不能自己说了算。

仲裁逻辑（两条，互相独立）：
  ① 参考氨基酸校验：我从序列翻译的 ref 密码子氨基酸 == HGVS 写的氨基酸？
     这一条**只检验我的序列提取与相位**，与"是否同义"无关。
     若这条通过率高 ⇒ 我的提取是对的；那么 205 条不一致就是 ClinVar 标注的问题。
  ② 同义标记比对：HGVS 的 '=' 比例 vs 我自己翻译判定的同义比例。

输出：逐条清单 + 汇总，落盘 wb_frozen/t1_hgvs_arbitration.json
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import re, json
import numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
n = len(df)
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
ref = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = np.array([str(s) for s in df["char_seq"].values])
cpos = df["cpos"].values.astype(int)
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
def rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s.upper()))
plus = np.array([(len(s) > 45 and s[45].upper() == r) for s, r in zip(seqs, ref)])

bases = "TCAG"; aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CT = {}; i = 0
for a in bases:
    for b in bases:
        for c in bases:
            CT[a + b + c] = aas[i]; i += 1

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U", "Xaa": "X"}

PAT = re.compile(r"p\.([A-Za-z]{3})(\d+)(=?)([A-Za-z]{3}|Ter|\*)?")

rows = []
n_hgvs = 0
for i in range(n):
    nm = str(df["Name"].iloc[i])
    m = PAT.search(nm)
    if not m:
        continue
    n_hgvs += 1
    s = seqs[i].upper()
    if not plus[i]:
        s = rc(s)
    vi = int(vix[i]); off = int((cpos[i] - 1) % 3); st = vi - off
    if st < 0 or st + 3 > len(s):
        continue
    cod = s[st:st + 3]
    acod = cod[:off] + alt[i] + cod[off + 1:]
    my_ref_aa = CT.get(cod, "X")
    my_alt_aa = CT.get(acod, "X")
    hgvs_aa = AA3.get(m.group(1), "?")
    hgvs_is_syn = (m.group(3) == "=")
    my_is_syn = (my_ref_aa == my_alt_aa)
    rows.append((i, nm, cod, acod, my_ref_aa, my_alt_aa, hgvs_aa,
                 int(m.group(2)), hgvs_is_syn, my_is_syn))

R = pd.DataFrame(rows, columns=["idx", "name", "ref_codon", "alt_codon",
                                "my_ref_aa", "my_alt_aa", "hgvs_aa",
                                "aa_pos", "hgvs_syn", "my_syn"])
print(f"可解析 HGVS 的条数: {n_hgvs}/{n}")

# ① 参考氨基酸校验（只检验我的提取与相位，与同义与否无关）
agree = (R["my_ref_aa"] == R["hgvs_aa"]).sum()
print(f"\n① 我从序列翻译的 ref 氨基酸 == HGVS 写的氨基酸: "
      f"{agree}/{len(R)} = {agree/len(R):.4f}")
if agree / len(R) < 0.9:
    print("   ⇒ ❌ 我的序列提取或相位有问题，先修它，别急着怪数据")
else:
    print("   ⇒ ✅ 我的提取是对的")

# ② 同义标记比对
print(f"\n② HGVS 标为同义(=): {R['hgvs_syn'].sum()}/{len(R)} = {R['hgvs_syn'].mean():.4f}")
print(f"   我翻译判定为同义:   {R['my_syn'].sum()}/{len(R)} = {R['my_syn'].mean():.4f}")
conflict = R[R["hgvs_syn"] & (~R["my_syn"])]
print(f"   🔴 冲突（HGVS 说同义、我翻译说不是）: {len(conflict)} 条 "
      f"({len(conflict)/len(R):.4f})")

# 冲突条的性质
if len(conflict):
    print("\n   冲突条目样例（前 10）:")
    for _, r in conflict.head(10).iterrows():
        print(f"     {r['name'][:46]:48s} cod {r['ref_codon']}->{r['alt_codon']} "
              f"{r['my_ref_aa']}->{r['my_alt_aa']}  HGVS={r['hgvs_aa']}{r['aa_pos']}")
    # 冲突条中，我的 ref 氨基酸是否与 HGVS 一致？
    c_agree = (conflict["my_ref_aa"] == conflict["hgvs_aa"]).sum()
    print(f"\n   冲突条中我的 ref 氨基酸仍与 HGVS 一致: {c_agree}/{len(conflict)} "
          f"= {c_agree/max(1,len(conflict)):.4f}")
    print("   （若这条很高 ⇒ 我的 ref 密码子提取没错，是 alt 密码子/标注的问题）")
    # 冲突条中 alt 是否产生终止密码子
    nstop = (conflict["my_alt_aa"] == "*").sum()
    print(f"   冲突条中我翻译出终止密码子的: {nstop} 条")

json.dump({
    "n_hgvs_parsed": int(n_hgvs),
    "ref_aa_agree": {"n": int(agree), "frac": float(agree / len(R))},
    "hgvs_syn_frac": float(R["hgvs_syn"].mean()),
    "my_syn_frac": float(R["my_syn"].mean()),
    "conflict_n": int(len(conflict)),
    "conflict_frac": float(len(conflict) / len(R)),
    "conflict_ref_aa_agree": {"n": int((conflict["my_ref_aa"] == conflict["hgvs_aa"]).sum()),
                              "frac": float((conflict["my_ref_aa"] == conflict["hgvs_aa"]).mean())},
    "conflict_nonsense": int((conflict["my_alt_aa"] == "*").sum()),
}, open(f"{OUT}/t1_hgvs_arbitration.json", "w"), indent=2, ensure_ascii=False)
R.to_csv(f"{OUT}/t1_hgvs_arbitration.tsv", sep="\t", index=False)
print("\n已写", f"{OUT}/t1_hgvs_arbitration.json / .tsv")
