import sys, os, json, re, pandas as pd
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path

DATA_DIR = Path("./data")
CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3)), m.group(4), m.group(5)) if m else (None, None, None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos<1 or cpos>len(cds): return None
    ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
    c=dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

cds_cache = json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
raw = pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",low_memory=False)
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
p = v["Name"].apply(parse_hgvs)
v["tx_id"]=p.apply(lambda x:x[0]); v["cpos"]=p.apply(lambda x:x[1])
v["ref_allele"]=p.apply(lambda x:x[2]); v["alt_allele"]=p.apply(lambda x:x[3])
h = v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds_cache.keys()))
var = v[h].copy()

syn = var[var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))].copy()
mis = var[~var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))].copy()

print(f"Total synonymous: {len(syn)} (pathogenic={int(syn['label'].sum())}, benign={int(len(syn)-syn['label'].sum())})")
print(f"Total missense: {len(mis)} (pathogenic={int(mis['label'].sum())}, benign={int(len(mis)-mis['label'].sum())})")

print("\n=== Top Pathogenic Synonymous Variants (with review status) ===")
path_syn = syn[syn["label"]==1].copy()
path_syn = path_syn.sort_values("ReviewStatus", ascending=False)
for _, row in path_syn.head(20).iterrows():
    cds = cds_cache.get(row["tx_id"], "")
    codon_str = build_codon_str(cds, int(row["cpos"]))
    if codon_str is None: continue
    ci = (int(row["cpos"])-1)//3
    codon_at_pos = dna_to_codons(cds)[ci] if ci < len(dna_to_codons(cds)) else "?"
    print(f"  {row['Name'][:60]:60s} | RS={str(row['ReviewStatus']):15s} | Gene={str(row.get('GeneSymbol','?'))[:10]:10s} | cpos={int(row['cpos']):5d} | codon={codon_at_pos}")

print("\n=== Top Benign Synonymous Variants (with review status) ===")
ben_syn = syn[syn["label"]==0].copy()
ben_syn = ben_syn.sort_values("ReviewStatus", ascending=False)
for _, row in ben_syn.head(20).iterrows():
    cds = cds_cache.get(row["tx_id"], "")
    codon_str = build_codon_str(cds, int(row["cpos"]))
    if codon_str is None: continue
    ci = (int(row["cpos"])-1)//3
    codon_at_pos = dna_to_codons(cds)[ci] if ci < len(dna_to_codons(cds)) else "?"
    print(f"  {row['Name'][:60]:60s} | RS={str(row['ReviewStatus']):15s} | Gene={str(row.get('GeneSymbol','?'))[:10]:10s} | cpos={int(row['cpos']):5d} | codon={codon_at_pos}")

print("\n=== Well-known pathogenic synonymous variants (literature) ===")
known_genes = ["BRCA1","BRCA2","TP53","CFTR","MLH1","MSH2","EGFR","KRAS","BRAF","APC"]
for gene in known_genes:
    gene_var = syn[(syn["label"]==1) & (syn.get("GeneSymbol","").astype(str).str.contains(gene, na=False))]
    if len(gene_var) > 0:
        best = gene_var.sort_values("ReviewStatus", ascending=False).iloc[0]
        print(f"  {gene}: {best['Name'][:70]} | RS={best['ReviewStatus']}")