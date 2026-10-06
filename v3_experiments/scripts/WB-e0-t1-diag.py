# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-t1-diag.py (2026-10-03)
定位"我从序列翻译的 ref 氨基酸只有 53.45% 与 HGVS 吻合"的根因。

候选假设（逐条证伪，不猜）：
  H1 变异位置 vi 不是窗口中心（vix 坐标系与 char_seq 不同）
  H2 反向互补后索引要重算（对长度 91 应为 90-vi，只有 vi=45 时不变）
  H3 相位不该用 (cpos-1)%3，cpos 与 CDS 起点有偏移
  H4 char_seq 窗口本身取错（例如取的是基因组而非转录本、或未剪接）
  H5 链向判定 plus/minus 判反了

判据：对每条变异，检查
      s_win[vi] == ref（正链）或 COMP(ref)（负链）
  以及用 HGVS 的**参考氨基酸**反推密码子，看序列里能否找到。
"""
import os  # [脱敏] 供读取 CODONBENCH_EXP_ROOT
import numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"

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

print("H1: 变异位置与窗口中心")
L = np.array([len(s) for s in seqs])
cent = (L - 1) // 2
print(f"  vix 众数 {pd.Series(vix).mode()[0]}；窗口长度众数 {pd.Series(L).mode()[0]}")
print(f"  vix == (len-1)//2 的比例: {(vix == cent).mean():.4f}")
print(f"  窗口长度分布: {pd.Series(L).value_counts().head(4).to_dict()}")

print("\nH5: 链向判定")
plus = np.array([(len(s) > 45 and s[45].upper() == r) for s, r in zip(seqs, ref)])
minus = np.array([(len(s) > 45 and s[45].upper() == COMP.get(r, "N"))
                  for s, r in zip(seqs, ref)])
print(f"  plus={plus.sum()} minus={minus.sum()} 不可判={(~plus & ~minus).sum()}")

print("\nH1/H2: 取窗口后 s[vi] 是否等于 ref（正链）或 COMP(ref)（负链）")
tot = 0; okpos = 0
for i in range(n):
    s = seqs[i].upper(); vi = int(vix[i])
    if vi >= len(s):
        continue
    tot += 1
    if plus[i] and s[vi] == ref[i]:
        okpos += 1
    elif minus[i] and s[vi] == COMP.get(ref[i], "N"):
        okpos += 1
print(f"  s[vi] 与 ref（按链向）相符: {okpos}/{tot} = {okpos/tot:.4f}")
# 反向互补后，位置 90-vi 处是否等于 ref
okrc = 0
for i in range(n):
    s = rc(seqs[i]); vi = int(vix[i])
    if vi >= len(s):
        continue
    if minus[i] and s[len(s) - 1 - vi] == ref[i]:
        okrc += 1
print(f"  负链条 rc 后 s[len-1-vi] == ref: {okrc}/{int(minus.sum())} = "
      f"{okrc/max(1,int(minus.sum())):.4f}")

print("\nH3: 相位。用 HGVS 的参考氨基酸检验 (cpos-1)%3 是否给出正确密码子")
bases = "TCAG"; aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CT = {}; i = 0
for a in bases:
    for b in bases:
        for c in bases:
            CT[a + b + c] = aas[i]; i += 1
import re
AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
       "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
       "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
       "Tyr": "Y", "Val": "V", "Ter": "*"}
PAT = re.compile(r"p\.([A-Za-z]{3})(\d+)")
hgvs_aa = []
for nm in df["Name"]:
    m = PAT.search(str(nm))
    hgvs_aa.append(AA3.get(m.group(1), "?") if m else "?")
hgvs_aa = np.array(hgvs_aa)

def agree_rate(off_fn, tag, use_rc_pos=False):
    ok = 0; tot = 0
    for i in range(n):
        if hgvs_aa[i] == "?":
            continue
        s = seqs[i].upper()
        vi_raw = int(vix[i])
        if minus[i]:
            s = rc(s)
            vi = len(s) - 1 - vi_raw if use_rc_pos else vi_raw
        else:
            vi = vi_raw
        off = int(off_fn[i])
        st = vi - off
        if st < 0 or st + 3 > len(s):
            continue
        cod = s[st:st + 3]
        tot += 1
        if CT.get(cod, "X") == hgvs_aa[i]:
            ok += 1
    print(f"  {tag:34s} {ok}/{tot} = {ok/max(1,tot):.4f}")

off_cpos = (cpos - 1) % 3
agree_rate(off_cpos, "off=(cpos-1)%3, vi 不变")
agree_rate(off_cpos, "off=(cpos-1)%3, 负链用 len-1-vi", use_rc_pos=True)
for k in (0, 1, 2):
    agree_rate(np.full(n, k), f"固定 off={k}, vi 不变")
for k in (0, 1, 2):
    agree_rate(np.full(n, k), f"固定 off={k}, 负链用 len-1-vi", use_rc_pos=True)

print("\n诊断样例（前 5 条 HGVS 不吻合者）")
shown = 0
for i in range(n):
    if hgvs_aa[i] == "?" or shown >= 5:
        continue
    s = seqs[i].upper()
    vi = int(vix[i]); off = int(off_cpos[i]); st = vi - off
    if st < 0 or st + 3 > len(s):
        continue
    cod = s[st:st + 3]
    if CT.get(cod, "X") == hgvs_aa[i]:
        continue
    shown += 1
    print(f"  idx={i} name={str(df['Name'].iloc[i])[:44]}")
    print(f"    len={len(s)} vi={vi} cpos={cpos[i]} off={off} st={st} "
          f"plus={plus[i]} minus={minus[i]}")
    print(f"    s[vi]={s[vi]} ref={ref[i]} alt={alt[i]}   cod={cod} "
          f"翻译={CT.get(cod,'X')} HGVS={hgvs_aa[i]}")
    # 在该窗口里搜索能翻译出 HGVS 氨基酸且 off 位置 == ref 的密码子
    s2 = rc(s) if minus[i] else s
    vi2 = len(s2) - 1 - vi if minus[i] else vi
    found = []
    for st2 in range(max(0, vi2 - 2), min(len(s2) - 3, vi2) + 1):
        c2 = s2[st2:st2 + 3]
        o2 = vi2 - st2
        if 0 <= o2 < 3 and c2[o2] == ref[i] and CT.get(c2, "X") == hgvs_aa[i]:
            found.append((st2, o2, c2))
    print(f"    窗口内能翻译出 {hgvs_aa[i]} 且 off 位为 ref 的密码子: {found}")
