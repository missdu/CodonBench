import sys, os, json, re, pandas as pd
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path

DATA_DIR = Path("./data")
CODON_CONTEXT = 16

cds_cache = json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))

cols = ["Name","Type","ClinicalSignificance","ReviewStatus","GeneSymbol","ReferenceAlleleVCF","AlternateAlleleVCF"]
raw = pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",usecols=cols,low_memory=False)
snv = raw[raw["Type"]=="single nucleotide variant"].copy()

pk = ["Pathogenic","Likely pathogenic"]
bk = ["Benign","Likely benign"]
def classify(cs):
    cs = str(cs)
    if any(k in cs for k in pk): return 1
    if any(k in cs for k in bk): return 0
    return -1

snv["label"] = snv["ClinicalSignificance"].apply(classify)
v = snv[snv["label"]>=0].copy()
v = v[(v["ReferenceAlleleVCF"].str.len()==1)&(v["AlternateAlleleVCF"].str.len()==1)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

p = v["Name"].apply(parse_hgvs)
v["tx_id"]=p.apply(lambda x:x[0]); v["cpos"]=p.apply(lambda x:x[1])
h = v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds_cache.keys()))
var = v[h].copy()

syn = var[var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))].copy()

print(f"Total synonymous with CDS: {len(syn)} (path={int(syn['label'].sum())}, ben={int(len(syn)-syn['label'].sum())})")

print("\n=== Pathogenic Synonymous (by review status) ===")
path_syn = syn[syn["label"]==1].sort_values("ReviewStatus", ascending=False)
for i, (_, row) in enumerate(path_syn.head(15).iterrows()):
    print(f"  {i+1}. {row['Name'][:65]} | RS={str(row['ReviewStatus'])[:12]} | Gene={str(row.get('GeneSymbol','?'))[:10]}")

print("\n=== Benign Synonymous (by review status) ===")
ben_syn = syn[syn["label"]==0].sort_values("ReviewStatus", ascending=False)
for i, (_, row) in enumerate(ben_syn.head(15).iterrows()):
    print(f"  {i+1}. {row['Name'][:65]} | RS={str(row['ReviewStatus'])[:12]} | Gene={str(row.get('GeneSymbol','?'))[:10]}")

print("\n=== Known disease genes with pathogenic synonymous ===")
for gene in ["BRCA1","BRCA2","TP53","CFTR","MLH1","MSH2","EGFR","KRAS","BRAF","APC","DMD","F8","HEMOPHILIA"]:
    gv = syn[(syn["label"]==1) & (syn.get("GeneSymbol","").astype(str).str.contains(gene, na=False))]
    if len(gv) > 0:
        best = gv.sort_values("ReviewStatus", ascending=False).iloc[0]
        print(f"  {gene}: n={len(gv)}, best={best['Name'][:60]} | RS={best['ReviewStatus']}")