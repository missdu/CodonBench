#!/usr/bin/env python3
# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-frozen-v2.py (2026-10-03)

两件事，一次做完（它们共用同一次变异级解析）：
  A. 重建冻结包 v2 —— v1 的 codon/aa 字段是 EXP-018 修复**之前**的坏配方产物
  B. S4 剪接审计 —— 算「变异到最近外显子—内含子接点的距离」（Reject §二.4.1）

为什么 v1 必须废：
  冻结包 dataset_v1 生成于 17:03；链向/相位修复发生在 17:10–17:26。
  外部基准（HGVS 的 Name 列）核对：
    变异 idx0 = NM_031885.5 c.1797，HGVS 写 p.Lys599= ⇒ 参考密码子应译 K
    v1 写的是 CCT/P    ← 错
  这是**被测对象之外的判据**，不是自证。

外显子数据来源：UCSC refGene（hg19/hg38）。
  服务器无外网（实测 NCBI / Ensembl 均返回 000），故在本地下载后上传。

距离定义（写死，供稿内方法节引用）：
  pos0 = position_vcf - 1                （VCF 1-based → UCSC 0-based）
  外显子占 [start, end)（0-based 半开）
  d_exon = min over 所有外显子 of min(|pos0 - start|, |pos0 - (end-1)|)
      ⇒ d_exon = 0 表示变异就落在外显子的第一个或最后一个碱基上
  d_cds  = min(|pos0 - cdsStart|, |pos0 - (cdsEnd - 1)|)
      ⇒ 这就是原稿"CDS 边界 3nt 内"那一统计所用的量

判据（能证伪的才写）：
  P1 外部基准：翻译出的参考氨基酸必须吻合 HGVS 的 p. 表达 ⇒ 要求 ≥99%
  P2 负例甲：故意用**错的版本组**去查表（GRCh37 的变异查 hg38），
             匹配率和距离分布必须明显变差；若无差别，说明这个字段根本没接上
  P3 负例乙：随机配对转录本 ⇒ 距离应显著变大（右移），否则计算没生效

输出（永不覆盖）：
  results/supplementary/wb_rerun/wb_frozen/dataset_v2/variants.tsv
  results/supplementary/wb_rerun/wb_frozen/dataset_v2/dataset_meta.json
  results/supplementary/wb_rerun/wb_frozen/s4_splice_audit_v1.json
"""
import os
import re
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen"
V2 = f"{OUT}/dataset_v2"
REFGENE = f"{BASE}/WB-refgene_exons.tsv"
os.makedirs(V2, exist_ok=True)

# ---------------- 1. 读数据 ----------------
df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
n = len(df)
y = df["label"].values.astype(int)
print(f"[data] 样本 {n}, 基因 {len(pd.unique(df['gene_symbol']))}, "
      f"正样本 {y.sum()} ({y.mean():.3f})", flush=True)

# ---------------- 2. 遗传密码 ----------------
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON2AA = {}
_i = 0
for _x in _b:
    for _y in _b:
        for _z in _b:
            CODON2AA[_x + _y + _z] = _aa[_i]
            _i += 1

COMP_R = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}

# ---------------- 3. 链向：以 HGVS 为编码链基准 ----------------
ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = df["char_seq"].values

_PATC = re.compile(r"c\.(\d+)([ACGT])>([ACGT])")
_PATP = re.compile(r"p\.([A-Za-z]{3})(\d+)")
_AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
        "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
        "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
        "Tyr": "Y", "Val": "V", "Ter": "*"}

hg_ref, hg_aa, hg_syn = [], [], []
for _nm in df["Name"]:
    _m = _PATC.search(str(_nm))
    hg_ref.append(_m.group(2) if _m else None)
    _p = _PATP.search(str(_nm))
    hg_aa.append(_AA3.get(_p.group(1), "?") if _p else "?")
    hg_syn.append(bool(_p) and "=" in str(_nm).split("p.")[-1])
hg_ref = np.array(hg_ref, dtype=object)
hg_aa = np.array(hg_aa, dtype=object)

strand_plus = np.array([(hg_ref[i] is not None and hg_ref[i] == ref_base[i])
                        for i in range(n)])
strand_minus = np.array([(hg_ref[i] is not None
                          and hg_ref[i] == COMP_R.get(ref_base[i], "N"))
                         for i in range(n)])
strand_known = strand_plus | strand_minus
print(f"[strand] plus={int(strand_plus.sum())} minus={int(strand_minus.sum())} "
      f"不可判={int((~strand_known).sum())}", flush=True)

# ---------------- 4. 密码子解析（修复后配方） ----------------
# off 由 **CDS 位置** 决定，不是由序列内偏移决定
# char_seq 已是编码链方向 ⇒ 序列不动；负链只把 ref/alt 碱基取互补
vix_a = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
cpos_i = df["cpos"].values.astype(int)
off_arr = (cpos_i - 1) % 3

ref_codon, alt_codon, ref_aa, alt_aa, parse_ok = [], [], [], [], []
for gi in range(n):
    s = str(seqs[gi]).upper()
    vi = int(vix_a[gi])
    off = int(off_arr[gi])
    start = vi - off
    cod = s[start:start + 3]
    if len(cod) != 3 or not (0 <= off < 3):
        ref_codon.append("NNN"); alt_codon.append("NNN")
        ref_aa.append("?"); alt_aa.append("?"); parse_ok.append(False)
        continue
    rb = ref_base[gi] if strand_plus[gi] else COMP_R.get(ref_base[gi], "N")
    ab = alt_base[gi] if strand_plus[gi] else COMP_R.get(alt_base[gi], "N")
    if cod[off] != rb:
        ref_codon.append("NNN"); alt_codon.append("NNN")
        ref_aa.append("?"); alt_aa.append("?"); parse_ok.append(False)
        continue
    acod = cod[:off] + ab + cod[off + 1:]
    ref_codon.append(cod); alt_codon.append(acod)
    ref_aa.append(CODON2AA.get(cod, "X")); alt_aa.append(CODON2AA.get(acod, "X"))
    parse_ok.append(True)

parse_ok = np.array(parse_ok)
print(f"[codon] 解析成功 {int(parse_ok.sum())}/{n} "
      f"= {100*parse_ok.mean():.2f}%", flush=True)

# ---- P1 判据：外部基准（HGVS 的参考氨基酸）----
_cmp = [i for i in range(n) if parse_ok[i] and hg_aa[i] != "?"]
_match = sum(1 for i in _cmp if ref_aa[i] == hg_aa[i])
t1_rate = _match / max(1, len(_cmp))
print(f"[P1] 参考氨基酸 vs HGVS：吻合 {_match}/{len(_cmp)} = {100*t1_rate:.2f}%",
      flush=True)

_syn_ok = sum(1 for i in range(n) if parse_ok[i] and ref_aa[i] == alt_aa[i])
print(f"[P1] 翻译前后氨基酸一致 {_syn_ok}/{int(parse_ok.sum())} = "
      f"{100*_syn_ok/max(1,int(parse_ok.sum())):.2f}%"
      f"（不一致 {int(parse_ok.sum())-_syn_ok} 条）", flush=True)

# ---------------- 5. 外显子结构 ----------------
rg = pd.read_csv(REFGENE, sep="\t", dtype={"tx_acc": str, "chrom": str})
print(f"[refgene] 载入 {len(rg)} 行", flush=True)
# 排除 alt / random / hap / fix 补丁区，只留主染色体
rg = rg[~rg["chrom"].str.contains("_", na=False)]
print(f"[refgene] 去掉补丁区后 {len(rg)} 行", flush=True)

BUILD_MAP = {"GRCh37": "hg19", "GRCh38": "hg38"}
gb = df["Assembly"].values  # 注意：列名是 Assembly，不是 genome_build
if gb is None:
    raise SystemExit("缺 genome_build 列，无法按版本匹配（build 混用已在前序审计中确认）")
want_build = np.array([BUILD_MAP.get(str(x), "?") for x in gb], dtype=object)

tx_acc = np.array([str(x).split(".")[0] for x in df["tx_id"].values], dtype=object)
chrom_q = np.array(["chr" + str(c) for c in df["Chromosome"].values], dtype=object)
pos0 = df["PositionVCF"].values.astype(int) - 1

# 按 (acc, build) 建索引；同 key 多行时优先 chrom 与查询一致的
idx = {}
multi = 0
for row in rg.itertuples(index=False):
    key = (row.tx_acc, row.build)
    if key in idx:
        multi += 1
        continue
    idx[key] = row
print(f"[refgene] 索引 {len(idx)} 个 (acc,build)；同 key 多行被跳过的 {multi} 个", flush=True)


def edge_dist(p0, starts, ends):
    """到最近外显子边缘碱基的距离（0 = 就落在外显子首/末碱基上）。"""
    best = None
    for a, b in zip(starts, ends):
        for d in (abs(p0 - a), abs(p0 - (b - 1))):
            if best is None or d < best:
                best = d
    return best


d_exon = np.full(n, -1, dtype=int)
d_cds = np.full(n, -1, dtype=int)
n_exon = np.full(n, -1, dtype=int)
in_exon = np.zeros(n, dtype=bool)
matched = np.zeros(n, dtype=bool)

for i in range(n):
    key = (tx_acc[i], want_build[i])
    row = idx.get(key)
    if row is None:
        continue
    if row.chrom != chrom_q[i]:
        continue
    matched[i] = True
    starts = [int(x) for x in str(row.exon_starts).split(",")]
    ends = [int(x) for x in str(row.exon_ends).split(",")]
    d_exon[i] = edge_dist(int(pos0[i]), starts, ends)
    d_cds[i] = min(abs(int(pos0[i]) - int(row.cds_start)),
                   abs(int(pos0[i]) - (int(row.cds_end) - 1)))
    n_exon[i] = int(row.n_exon)
    in_exon[i] = any(a <= int(pos0[i]) < b for a, b in zip(starts, ends))

print(f"[exon] 匹配上转录本结构的 {int(matched.sum())}/{n} = "
      f"{100*matched.mean():.2f}%；落在注释外显子内的 {int(in_exon.sum())}", flush=True)

# ---------------- 6. 负例 ----------------
# P2：用错的版本组去查（GRCh37 的变异查 hg38，反之亦然）
wrong_build = np.array(["hg38" if b == "hg19" else "hg19" for b in want_build],
                       dtype=object)
m_wrong = np.zeros(n, dtype=bool)
d_wrong = np.full(n, -1, dtype=int)
for i in range(n):
    row = idx.get((tx_acc[i], wrong_build[i]))
    if row is None or row.chrom != chrom_q[i]:
        continue
    m_wrong[i] = True
    starts = [int(x) for x in str(row.exon_starts).split(",")]
    ends = [int(x) for x in str(row.exon_ends).split(",")]
    d_wrong[i] = edge_dist(int(pos0[i]), starts, ends)
print(f"[P2 负例] 用错版本组：匹配 {int(m_wrong.sum())}（正确版 {int(matched.sum())}）；"
      f"两者都匹配上时距离中位数 "
      f"正确={np.median(d_exon[matched & m_wrong]) if (matched & m_wrong).any() else 'NA'} "
      f"错误={np.median(d_wrong[matched & m_wrong]) if (matched & m_wrong).any() else 'NA'}",
      flush=True)

# P3：随机配对转录本
rng = np.random.default_rng(20261003)
perm = rng.permutation(n)
d_perm = np.full(n, -1, dtype=int)
m_perm = np.zeros(n, dtype=bool)
for i in range(n):
    j = perm[i]
    row = idx.get((tx_acc[j], want_build[j]))
    if row is None:
        continue
    m_perm[i] = True
    starts = [int(x) for x in str(row.exon_starts).split(",")]
    ends = [int(x) for x in str(row.exon_ends).split(",")]
    d_perm[i] = edge_dist(int(pos0[i]), starts, ends)
both = matched & m_perm
print(f"[P3 负例] 随机配转录本：距离中位数 "
      f"真实={np.median(d_exon[both]) if both.any() else 'NA'} vs "
      f"随机={np.median(d_perm[both]) if both.any() else 'NA'}", flush=True)

# ---------------- 7. S4 统计 ----------------
m = matched
d = d_exon[m]
dc = d_cds[m]
lab = y[m]


def _hist(arr, thr):
    return {f"<={t}": int((arr <= t).sum()) for t in thr}


def _frac(arr, thr):
    return {f"<={t}": round(float((arr <= t).mean()), 6) for t in thr}


THR = [0, 1, 2, 3, 8]
s4 = {
    "n_total": n,
    "n_matched": int(m.sum()),
    "match_rate": round(float(m.mean()), 6),
    "n_in_exon": int(in_exon.sum()),
    "n_exon_per_tx": {
        "median": float(np.median(n_exon[m])),
        "mean": round(float(n_exon[m].mean()), 3),
        "min": int(n_exon[m].min()),
        "max": int(n_exon[m].max()),
    },
    "d_exon_edge": {
        "min": int(d.min()), "median": float(np.median(d)),
        "mean": round(float(d.mean()), 3), "max": int(d.max()),
        "count_by_thr": _hist(d, THR), "frac_by_thr": _frac(d, THR),
    },
    "d_cds_edge": {
        "min": int(dc.min()), "median": float(np.median(dc)),
        "mean": round(float(dc.mean()), 3), "max": int(dc.max()),
        "count_by_thr": _hist(dc, THR), "frac_by_thr": _frac(dc, THR),
    },
    "by_label": {},
    "negative_controls": {
        "P2_wrong_build": {
            "n_matched": int(m_wrong.sum()),
            "median_dist_when_both": (float(np.median(d_wrong[matched & m_wrong]))
                                      if (matched & m_wrong).any() else None),
            "median_dist_correct": (float(np.median(d_exon[matched & m_wrong]))
                                    if (matched & m_wrong).any() else None),
        },
        "P3_shuffled_tx": {
            "median_dist_real": float(np.median(d_exon[both])) if both.any() else None,
            "median_dist_shuffled": float(np.median(d_perm[both])) if both.any() else None,
        },
    },
}
for lb, nm in ((0, "benign"), (1, "pathogenic")):
    sel = lab == lb
    s4["by_label"][nm] = {
        "n": int(sel.sum()),
        "d_exon_frac_by_thr": _frac(d[sel], THR),
        "d_cds_frac_by_thr": _frac(dc[sel], THR),
        "median_d_exon": float(np.median(d[sel])),
        "median_d_cds": float(np.median(dc[sel])),
    }

print("\n=== S4 剪接审计 ===")
print(json.dumps(s4, ensure_ascii=False, indent=2)[:3000], flush=True)

# ---------------- 8. 落盘 ----------------
out = pd.DataFrame({
    "allele_id": df["#AlleleID"].values if "#AlleleID" in df.columns else np.arange(n),
    "variation_id": df["VariationID"].values if "VariationID" in df.columns else "",
    "genome_build": gb,
    "chromosome": df["Chromosome"].values,
    "position_vcf": df["PositionVCF"].values,
    "ref": ref_base, "alt": alt_base,
    "gene_id": df["gene_symbol"].values,
    "gene_id_raw": df["gene_symbol"].values,
    "transcript_id_with_version": df["tx_id"].values,
    "consequence": "synonymous (SynPath task definition)",
    "reference_codon": ref_codon, "alternate_codon": alt_codon,
    "reference_aa": ref_aa, "alternate_aa": alt_aa,
    "cds_position": cpos_i,
    "exon_boundary_distance": np.where(matched, d_exon, "MISSING"),
    "cds_boundary_distance": np.where(matched, d_cds, "MISSING"),
    "n_exon_in_tx": np.where(matched, n_exon, "MISSING"),
    "clinvar_label": y,
    "review_status": df["ReviewStatus"].values,
    "clinvar_release": "MISSING",
    "strand": np.where(strand_plus, "+", np.where(strand_minus, "-", "?")),
    "codon_parse_ok": parse_ok,
    "seq_len": [len(str(s)) for s in seqs],
})
fold_npz = np.load(f"{OUT}/folds_240logo_v1.npz", allow_pickle=True)
fmap = dict(zip(fold_npz["global_index"], fold_npz["fold_id"])) \
    if "global_index" in fold_npz.files else None
if fmap is not None:
    out["logo_fold_id"] = [fmap.get(int(i), -1) for i in range(n)]
out.to_csv(f"{V2}/variants.tsv", sep="\t", index=False)

meta = {
    "batch_id": "dataset_v2",
    "created": "2026-10-03",
    "supersedes": "dataset_v1（codon/aa 字段为 EXP-018 修复前配方，已作废）",
    "n": n,
    "codon_recipe": {
        "off": "(cpos - 1) % 3   ← 相位由 CDS 位置决定",
        "strand": "以 HGVS Name 列的 c. 表达为编码链基准",
        "negative_strand": "char_seq 是编码链 ⇒ 序列不动；只把 ref/alt 碱基取互补",
    },
    "T1_external_arbitration": {
        "judge": "HGVS p. 表达给出的参考氨基酸（独立于本项目的翻译）",
        "match": _match, "denominator": len(_cmp), "rate": round(float(t1_rate), 6),
        "verdict": "PASS" if t1_rate >= 0.99 else "FAIL",
        "aa_identical_before_after": {
            "n": _syn_ok, "denominator": int(parse_ok.sum()),
            "rate": round(_syn_ok / max(1, int(parse_ok.sum())), 6),
        },
    },
    "exon_annotation": {
        "source": "UCSC refGene (hg19 + hg38)，本地下载后上传（服务器无外网）",
        "match_rate": round(float(m.mean()), 6),
        "note": "同一 accession 在同一版本出现多行时取第一条（主染色体）",
    },
    "genome_build_mix": {str(k): int(v) for k, v in pd.Series(gb).value_counts().items()},
    "MISSING_fields": ["clinvar_release"],
}
with open(f"{V2}/dataset_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, ensure_ascii=False, indent=2)
with open(f"{OUT}/s4_splice_audit_v1.json", "w", encoding="utf-8") as fh:
    json.dump(s4, fh, ensure_ascii=False, indent=2)

print(f"[out] {V2}/variants.tsv")
print(f"[out] {V2}/dataset_meta.json")
print(f"[out] {OUT}/s4_splice_audit_v1.json", flush=True)
