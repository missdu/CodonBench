# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB | E1 严格版 —— Phase A2：序列构造与自检（纯 CPU）

Phase A 的发现（已确认，本脚本据此修正）：
  * char_seq[45] 与参考碱基：正链匹配 1427、互补匹配 1403、无法判定 10
    ⇒ 负链基因的窗口取了反向互补，**构造 alt 必须先定链**，
      否则约一半样本的"变异碱基"放错（会变成错义/无义，直接毁掉整个 E1）
  * HGVS 的 p.Xxx###= 记号 2840/2840 = 100% ⇒ 数据集本身确实全是同义变异

本脚本产出（供 Phase B 提嵌入用）：
  e1s_ref.npy   —— 参考窗口
  e1s_alt.npy   —— 变异窗口（按链取互补后替换）
  e1s_pseudo.npy—— 假变异窗口：在**同一窗口的另一个位置**换一个碱基（阴性对照）
  e1s_meta.json —— 链向、相位、可用掩码

同时做两项自检：
  [框校验] 枚举 3 个阅读相位，看 ref/alt 翻译是否相等；
           统计"满足等价的相位数"分布，并用 cpos 推出的框做交叉验证
  [同义校验] 在满足等价的相位下，直接给出变异密码子与氨基酸
"""
import os
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("all_proxy", None)

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
CENTER = 45

_bases = "TCAG"
_aas = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODON = {}
_i = 0
for b1 in _bases:
    for b2 in _bases:
        for b3 in _bases:
            CODON[b1 + b2 + b3] = _aas[_i]
            _i += 1
COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def comp(b):
    return COMP.get(b, "N")


def revcomp(s):
    return "".join(comp(c) for c in reversed(s))


def translate(s):
    return "".join(CODON.get(s[i:i + 3], "X")
                   for i in range(0, len(s) - len(s) % 3, 3))


char_seq = df["char_seq"].astype(str).values
ref_base = df["ReferenceAlleleVCF"].astype(str).values
alt_base = df["AlternateAlleleVCF"].astype(str).values
cpos = df["cpos"].values.astype(float)
labels = df["label"].values
genes = df["gene_symbol"].values

# ---------- 1. 定链 ----------
strand = np.zeros(n, dtype=int)
for i, (s, r) in enumerate(zip(char_seq, ref_base)):
    if len(s) <= CENTER:
        continue
    if s[CENTER] == r:
        strand[i] = 1
    elif s[CENTER] == comp(r):
        strand[i] = -1
print("链向: 正链 %d | 负链 %d | 未定 %d"
      % ((strand == 1).sum(), (strand == -1).sum(), (strand == 0).sum()))

# ---------- 2. 转到编码链 ----------
# 负链的 char_seq 是反向互补 ⇒ 反向互补回来，变异位点从 45 变成 len-1-45
cod_ref, cix = [], []
for s, st in zip(char_seq, strand):
    if st == -1:
        cod_ref.append(revcomp(s))
        cix.append(len(s) - 1 - CENTER)
    else:
        cod_ref.append(s)
        cix.append(CENTER)
cix = np.array(cix)

# 在编码链上替换（此时直接用 VCF 的碱基，不再取互补）
cod_alt = []
for s, a, ix, st in zip(cod_ref, alt_base, cix, strand):
    if st == 0 or ix < 0 or ix >= len(s):
        cod_alt.append(s)
        continue
    cod_alt.append(s[:ix] + a + s[ix + 1:])

# ---------- 3. 相位枚举校验翻译等价性 ----------
nphase_ok = np.zeros(n, dtype=int)     # 3 个相位里满足 ref/alt 翻译相等的相位数
phase_pick = -np.ones(n, dtype=int)    # 第一个满足的相位
cpos_phase = ((cpos - 1) % 3).astype(int)
cpos_phase_ok = np.zeros(n, dtype=int)
# 变异位点在编码链窗口里的密码子起始 = cix - phase（相位定义为 cix 在密码子内的偏移）
for i, (sr, sa, ix) in enumerate(zip(cod_ref, cod_alt, cix)):
    for ph in range(3):
        start = ix - ph
        if start < 0 or len(sr) - start < 3:
            continue
        L = len(sr) - start
        L -= L % 3
        if translate(sr[start:start + L]) == translate(sa[start:start + L]):
            nphase_ok[i] += 1
            if phase_pick[i] < 0:
                phase_pick[i] = ph
    # cpos 推出的框：变异位点在密码子内的偏移 = (cpos-1)%3
    ph = int(cpos_phase[i]) if not np.isnan(cpos[i]) else -1
    if ph >= 0:
        start = ix - ph
        if start >= 0 and len(sr) - start >= 3:
            L = len(sr) - start
            L -= L % 3
            if translate(sr[start:start + L]) == translate(sa[start:start + L]):
                cpos_phase_ok[i] = 1

print("\n[框校验] 3 个阅读相位中满足 ref/alt 翻译相等的相位数分布:")
for k in range(4):
    print("   相位数 = %d : %d 条 (%.2f%%)" % (k, (nphase_ok == k).sum(),
                                              100 * (nphase_ok == k).mean()))
print("   至少 1 个相位满足: %d / %d (%.2f%%)"
      % ((nphase_ok >= 1).sum(), n, 100 * (nphase_ok >= 1).mean()))
print("[框校验] cpos 推出的相位满足等价: %d / %d (%.2f%%)"
      % (cpos_phase_ok.sum(), n, 100 * cpos_phase_ok.mean()))

# ---------- 4. 可用掩码 ----------
usable = (strand != 0) & (nphase_ok >= 1)
print("\n[可用掩码] 可定链 且 至少一个相位翻译等价: %d / %d (%.2f%%)"
      % (usable.sum(), n, 100 * usable.mean()))
drop_strand = int((strand == 0).sum())
drop_frame = int(((strand != 0) & (nphase_ok == 0)).sum())
print("   剔除: 链未定 %d | 无相位满足（可能非同义或框异常）%d"
      % (drop_strand, drop_frame))

# ---------- 5. 阴性对照：假变异窗口 ----------
# 在同一个窗口里，离变异位点 15 nt 远的位置换成一个不同的碱基
rng = np.random.RandomState(20261002)
cod_pseudo = []
for s, ix, st in zip(cod_ref, cix, strand):
    if st == 0 or ix < 0 or ix >= len(s):
        cod_pseudo.append(s)
        continue
    j = ix - 15 if ix - 15 >= 0 else ix + 15
    if j < 0 or j >= len(s):
        cod_pseudo.append(s)
        continue
    old = s[j]
    new = old
    for _ in range(8):
        new = rng.choice(list("ACGT"))
        if new != old:
            break
    cod_pseudo.append(s[:j] + new + s[j + 1:])

# ---------- 6. 落盘 ----------
np.save(OUT / "e1s_ref.npy", np.array(cod_ref, dtype=object), allow_pickle=True)
np.save(OUT / "e1s_alt.npy", np.array(cod_alt, dtype=object), allow_pickle=True)
np.save(OUT / "e1s_pseudo.npy", np.array(cod_pseudo, dtype=object), allow_pickle=True)
np.save(OUT / "e1s_strand.npy", strand)
np.save(OUT / "e1s_variant_index.npy", cix)
np.save(OUT / "e1s_phase_pick.npy", phase_pick)
np.save(OUT / "e1s_usable.npy", usable)
np.save(OUT / "e1s_labels.npy", labels)
np.save(OUT / "e1s_genes.npy", np.array(genes, dtype=object), allow_pickle=True)

meta = {
    "n_total": n,
    "n_usable": int(usable.sum()),
    "strand": {"plus": int((strand == 1).sum()),
               "minus": int((strand == -1).sum()),
               "unknown": int((strand == 0).sum())},
    "phase_equal_dist": {str(k): int((nphase_ok == k).sum()) for k in range(4)},
    "cpos_phase_equal": int(cpos_phase_ok.sum()),
    "pseudo_offset_nt": 15,
    "note": ("ref/alt/pseudo 均为**编码链**窗口（负链已反向互补回来），"
             "变异位点索引见 e1s_variant_index.npy；"
             "pseudo 为同一窗口内偏移 15 nt 处的单碱基替换，用作阴性对照"),
}
with open(OUT / "e1s_meta.json", "w") as f:
    json.dump(meta, f, indent=1, ensure_ascii=False)
print("\n已落盘:", sorted(p.name for p in OUT.glob("e1s_*")))

# ---------- 7. 抽样打印，肉眼可核 ----------
print("\n[抽样核对] 前 3 条（编码链，变异位点标 []）:")
for i in range(3):
    ix = cix[i]
    print("  #%d %s | 链 %+d | 相位 %d" % (
        i, str(df["Name"].iloc[i])[:38], strand[i], phase_pick[i]))
    print("     ref ...%s[%s]%s..." % (cod_ref[i][max(0, ix - 9):ix],
                                       cod_ref[i][ix],
                                       cod_ref[i][ix + 1:ix + 10]))
    print("     alt ...%s[%s]%s..." % (cod_alt[i][max(0, ix - 9):ix],
                                       cod_alt[i][ix],
                                       cod_alt[i][ix + 1:ix + 10]))
    ph = phase_pick[i] if phase_pick[i] >= 0 else 0
    st = ix - ph
    if st >= 0:
        cr = cod_ref[i][st:st + 3]
        ca = cod_alt[i][st:st + 3]
        print("     变异密码子 %s -> %s | 氨基酸 %s -> %s"
              % (cr, ca, CODON.get(cr, "X"), CODON.get(ca, "X")))
