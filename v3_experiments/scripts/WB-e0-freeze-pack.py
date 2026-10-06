# [脱敏] 本机服务器的绝对路径已全部移除，改为读取环境变量
#        CODONBENCH_EXP_ROOT      —— 实验数据根目录
#        CODONBENCH_SERVER_HOME   —— 服务器 home（conda 等安装路径用）
#        两个变量的含义与取值见仓库根目录 README.md。
"""
WB-e0-freeze-pack.py (2026-10-03)
建立 GPT E0 要求的**可冻结审计数据包**的数据层部分（模型层要等 GPU 提取）。

GPT 原文：数据清单每条变异至少包含 15 项。本脚本把能算的补齐、能查的登记，
查不到的**明确标记为 MISSING，不编造**。

目录（永不覆盖）：results/supplementary/wb_rerun/wb_frozen/dataset_v1/
  variants.tsv      每条变异的审计字段（E0 15 项 + 派生）
  dataset_meta.json 数据集级元数据（分布、缺失、去重规则、已知瑕疵）
  链接：folds_240logo_v1.npz（EXP-015 已冻结）、seqhash/（EXP-013）

🔴 本轮实测登记的两条瑕疵（必须写进 SI，不能藏）：
  1. genome_build 混合：GRCh37 = 1444 / GRCh38 = 1396
     缓解证据：链向判定 99.8% 成功 ⇒ 序列与参考碱基匹配，未造成错配
  2. char_seq 长度 < 91 的 55 条（1.9%），其中 35 条进入 240 folds
     ⇒ 这些变异的序列窗口被截断，上下文不完整
"""
import os, json, hashlib
import numpy as np, pandas as pd

BASE = os.environ.get("CODONBENCH_EXP_ROOT", "/path/to/your/codonbench-exp")
SYN = f"{BASE}/results/unified_eval/synpath_data"
SUPP = f"{BASE}/results/supplementary"
OUT = f"{SUPP}/wb_rerun/wb_frozen/dataset_v1"
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet(f"{SYN}/synpath_variants.parquet")
n = len(df)
print(f"样本 {n}, 列 {len(df.columns)}")

CENTER = 45
COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
CODON_TABLE = {}
bases = "TCAG"
aas = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
i = 0
for a in bases:
    for b in bases:
        for c in bases:
            CODON_TABLE[a + b + c] = aas[i]; i += 1

ref_base = np.array([str(x).upper() for x in df["ReferenceAlleleVCF"].values])
alt_base = np.array([str(x).upper() for x in df["AlternateAlleleVCF"].values])
seqs = np.array([str(s) for s in df["char_seq"].values])
vix = np.load(f"{SUPP}/wb_rerun/e1s_variant_index.npy")
ph = np.load(f"{SUPP}/wb_rerun/e1s_phase_pick.npy")

strand_plus = np.array([(len(s) > CENTER and s[CENTER].upper() == r)
                        for s, r in zip(seqs, ref_base)])
strand_minus = np.array([(len(s) > CENTER and s[CENTER].upper() == COMP.get(r, "N"))
                         for s, r in zip(seqs, ref_base)])
strand = np.where(strand_plus, "+", np.where(strand_minus, "-", "?"))

def _rc(s):
    return "".join(COMP.get(c, "N") for c in reversed(s.upper()))

rcod, acod, raa, aaa, ok = [], [], [], [], []
for i in range(n):
    s = seqs[i].upper()
    if not strand_plus[i]:
        s = _rc(s)
    vi = int(vix[i]); st = vi - ((vi - int(ph[i])) % 3)
    cod = s[st:st + 3]; off = vi - st
    if len(cod) != 3 or not (0 <= off < 3) or cod[off] != ref_base[i]:
        rcod.append(""); acod.append(""); raa.append(""); aaa.append(""); ok.append(False)
        continue
    a = cod[:off] + alt_base[i] + cod[off + 1:]
    rcod.append(cod); acod.append(a)
    raa.append(CODON_TABLE.get(cod, "X")); aaa.append(CODON_TABLE.get(a, "X"))
    ok.append(True)
ok = np.array(ok)
print(f"单密码子解析成功 {ok.sum()}/{n} = {100*ok.mean():.2f}%")
# T1：同义变异翻译前后氨基酸必须一致
same_aa = np.array([raa[i] == aaa[i] for i in range(n)])
print(f"T1 氨基酸一致（成功解析者中）: {same_aa[ok].mean():.4f}  "
      f"({same_aa[ok].sum()}/{ok.sum()})")

seq_hash = np.array([hashlib.sha256(s.encode()).hexdigest()[:16] for s in seqs])
alt_seqs = []
for i in range(n):
    s = seqs[i]
    vi = int(vix[i])
    alt_seqs.append(s[:vi] + alt_base[i] + s[vi + 1:])
alt_hash = np.array([hashlib.sha256(s.encode()).hexdigest()[:16] for s in alt_seqs])

d = np.load(f"{SUPP}/wb_rerun/wb_frozen/folds_240logo_v1.npz", allow_pickle=True)
fold_id = d["fold_id"]; gidx = d["global_index"]
in_fold = np.full(n, -1, dtype=int)
in_fold[gidx] = fold_id

tsv = pd.DataFrame({
    "variant_id": df["#AlleleID"].values,
    "variation_id": df["VariationID"].values,
    "genome_build": df["Assembly"].values,
    "chromosome": df["Chromosome"].values,
    "position_vcf": df["PositionVCF"].values,
    "ref": ref_base, "alt": alt_base,
    "gene_id": df["gene_symbol_hgvs"].values,          # 497 个，与 tx 一一对应
    "gene_id_raw": df["gene_symbol"].values,           # 498 个，含 1 个复合标签
    "transcript_id_with_version": df["tx_id"].values,
    "consequence": np.where(df["is_syn"].values, "synonymous", "OTHER"),
    "reference_codon": rcod, "alternate_codon": acod,
    "reference_aa": raa, "alternate_aa": aaa,
    "cds_position": df["cpos"].values,
    "exon_boundary_distance": "MISSING",               # 🔴 无外显子注释，Reject §二.4.1 点名
    "clinvar_label": df["label"].values,
    "review_status": df["ReviewStatus"].values,
    "clinvar_release": "MISSING",                      # 未登记，须补 LastEvaluated 或抓取日期
    "strand": strand,
    "codon_parse_ok": ok,
    "seq_len": [len(s) for s in seqs],
    "ref_seq_sha256_16": seq_hash,
    "alt_seq_sha256_16": alt_hash,
    "logo_fold_id": in_fold,                           # -1 = 未进入任何有效折
})
tsv.to_csv(f"{OUT}/variants.tsv", sep="\t", index=False)
print("已写", f"{OUT}/variants.tsv", tsv.shape)

L = np.array([len(s) for s in seqs])
meta = {
    "source": f"{SYN}/synpath_variants.parquet",
    "n_variants": int(n),
    "n_transcripts": int(df["tx_id"].nunique()),
    "n_genes_hgvs": int(df["gene_symbol_hgvs"].nunique()),
    "n_genes_raw": int(df["gene_symbol"].nunique()),
    "pos_rate": float(df["label"].mean()),
    "genome_build": {str(k): int(v) for k, v in df["Assembly"].value_counts().items()},
    "genome_build_in_folds": {str(k): int(v) for k, v in df.iloc[gidx]["Assembly"].value_counts().items()},
    "review_status": {str(k): int(v) for k, v in df["ReviewStatus"].value_counts().items()},
    "codon_parse_ok": {"n": int(ok.sum()), "frac": float(ok.mean())},
    "T1_aa_identical_among_parsed": {"n": int(same_aa[ok].sum()), "frac": float(same_aa[ok].mean())},
    "seq_len": {"modal": int(pd.Series(L).mode()[0]),
                "n_shorter_than_modal": int((L < pd.Series(L).mode()[0]).sum()),
                "frac_shorter": float((L < pd.Series(L).mode()[0]).mean()),
                "n_shorter_in_folds": int((L[gidx] < pd.Series(L).mode()[0]).sum())},
    "dedup_rule": {
        "gene_to_tx_multiplicity": int((df.groupby("gene_symbol")["tx_id"].nunique() > 1).sum()),
        "verdict": "0 ⇒ 没有任何基因有第二个转录本 ⇒ 留一转录本 ≡ 留一基因",
        "tx_to_gene_multiplicity": int((df.groupby("tx_id")["gene_symbol"].nunique() > 1).sum()),
        "known_defect": "NM_000548.5 的 gene_symbol 为 'PKD1;TSC2'（复合标签，3 行）"
                        "⇒ 分组建议用 gene_symbol_hgvs",
    },
    "MISSING_fields": ["exon_boundary_distance", "clinvar_release"],
    "linked_artifacts": {
        "folds": "wb_frozen/folds_240logo_v1.npz (EXP-015)",
        "seqhash": "wb_rerun/seqhash/*.sha256.npy (EXP-013)",
    },
}
json.dump(meta, open(f"{OUT}/dataset_meta.json", "w"), indent=2, ensure_ascii=False)
print("已写", f"{OUT}/dataset_meta.json")
print(json.dumps({k: v for k, v in meta.items()
                  if k in ("genome_build", "seq_len", "codon_parse_ok",
                           "T1_aa_identical_among_parsed", "MISSING_fields")},
                 indent=2, ensure_ascii=False))
