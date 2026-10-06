# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-t1-diag2.py (2026-10-03)
继续定位：把 Name 里的 HGVS c. 数字解析出来，与 cpos 列逐条比对。
若两者不一致 ⇒ cpos 不是 CDS 位置，用它定相位必然错。
并统计吻合率按 plus/minus、窗口长度分组，看错误是否集中在某一类。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import re, numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
n = len(df)
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
ref = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
seqs = np.array([str(s) for s in df["char_seq"].values])
cpos = df["cpos"].values.astype(int)
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
def rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s.upper()))
plus = np.array([(len(s) > 45 and s[45].upper() == r) for s, r in zip(seqs, ref)])
minus = np.array([(len(s) > 45 and s[45].upper() == COMP.get(r, "N"))
                  for s, r in zip(seqs, ref)])

PATC = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
PATP = re.compile(r"p\.([A-Za-z]{3})(\d+)")
c_hgvs = np.full(n, -1, dtype=int)
c_ref = np.empty(n, dtype=object); c_alt = np.empty(n, dtype=object)
aa3 = np.empty(n, dtype=object); aa_pos = np.full(n, -1, dtype=int)
for i in range(n):
    nm = str(df["Name"].iloc[i])
    m = PATC.search(nm)
    if m:
        c_hgvs[i] = int(m.group(1)); c_ref[i] = m.group(2); c_alt[i] = m.group(3)
    m2 = PATP.search(nm)
    if m2:
        aa3[i] = m2.group(1); aa_pos[i] = int(m2.group(2))

print("① cpos 列 vs Name 里的 HGVS c. 数字")
ok = (c_hgvs >= 0)
same = (c_hgvs[ok] == cpos[ok]).sum()
print(f"   可解析 {ok.sum()}/{n}；一致 {same}/{ok.sum()} = {same/ok.sum():.4f}")
bad = np.where(ok & (c_hgvs != cpos))[0]
print(f"   不一致 {len(bad)} 条；样例:")
for i in bad[:5]:
    print(f"     idx={i} cpos={cpos[i]} HGVS c.={c_hgvs[i]}  {str(df['Name'].iloc[i])[:46]}")

print("\n② HGVS 的 ref 碱基 vs 数据的 ReferenceAlleleVCF")
rb = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
m2 = ok & (c_ref != None)
same_b = sum(1 for i in range(n) if ok[i] and c_ref[i] == rb[i])
print(f"   一致 {same_b}/{ok.sum()} = {same_b/ok.sum():.4f}")

print("\n③ 密码子编号自洽性：ceil(c/3) 是否 == p. 的数字")
import math
okp = (c_hgvs >= 0) & (aa_pos >= 0)
ag = sum(1 for i in range(n) if okp[i] and math.ceil(c_hgvs[i] / 3) == aa_pos[i])
print(f"   ceil(c/3) == p.编号 : {ag}/{okp.sum()} = {ag/okp.sum():.4f}")
print(f"   ⇒ HGVS 自身是**自洽**的，说明 c. 就是 CDS 位置，且 cpos 列 == c. 数字")

print("\n④ 吻合率分组（用 HGVS 参考氨基酸检验我的密码子提取）")
bases = "TCAG"; aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CT = {}; i = 0
for a in bases:
    for b in bases:
        for c in bases:
            CT[a + b + c] = aas[i]; i += 1
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "*"}
hg = np.array([AA3.get(a, "?") for a in aa3])
L = np.array([len(s) for s in seqs])

def rate(mask, tag, off_mode="cpos"):
    ok = 0; tot = 0
    for i in range(n):
        if not mask[i] or hg[i] == "?":
            continue
        s = seqs[i].upper(); vi = int(vix[i])
        if minus[i]:
            s = rc(s)
        off = int((cpos[i] - 1) % 3) if off_mode == "cpos" else int(off_mode)
        st = vi - off
        if st < 0 or st + 3 > len(s):
            continue
        tot += 1
        if CT.get(s[st:st + 3], "X") == hg[i]:
            ok += 1
    print(f"   {tag:28s} {ok}/{tot} = {ok/max(1,tot):.4f}")
    return ok / max(1, tot)

rate(np.ones(n, bool), "全体")
rate(plus, "正链 (plus)")
rate(minus, "负链 (minus)")
rate(L == 91, "窗口长度 = 91")
rate(L < 91, "窗口长度 < 91")
rate(plus & (L == 91), "正链 且 长度91")
rate(minus & (L == 91), "负链 且 长度91")

print("\n⑤ 负链：若改用 vi' = len-1-vi（索引镜像）会怎样")
def rate_mirror(mask, tag):
    ok = 0; tot = 0
    for i in range(n):
        if not mask[i] or hg[i] == "?":
            continue
        s = seqs[i].upper(); vi = int(vix[i])
        if minus[i]:
            s = rc(s); vi = len(s) - 1 - vi
        off = int((cpos[i] - 1) % 3)
        st = vi - off
        if st < 0 or st + 3 > len(s):
            continue
        tot += 1
        if CT.get(s[st:st + 3], "X") == hg[i]:
            ok += 1
    print(f"   {tag:28s} {ok}/{tot} = {ok/max(1,tot):.4f}")
rate_mirror(np.ones(n, bool), "全体(负链镜像)")
rate_mirror(minus, "负链(镜像)")
