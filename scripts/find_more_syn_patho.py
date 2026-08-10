import pandas as pd
import json
import re
from pathlib import Path
from collections import Counter

data_dir = Path("./data")
clinvar_path = data_dir / "task2_clinvar" / "clinvar_raw.txt.gz"
cds_path = data_dir / "task2_clinvar" / "cds_sequences.json"

print("Loading CDS cache...")
cds = json.load(open(cds_path))
cds_txids = set(cds.keys())
print(f"CDS cache: {len(cds)} transcripts")

print("Loading FULL ClinVar (this may take a few minutes)...")
raw = pd.read_csv(clinvar_path, sep="\t", low_memory=False)
print(f"Total rows: {len(raw):,}")

snv = raw[raw["Type"] == "single nucleotide variant"].copy()
print(f"SNVs: {len(snv):,}")

patho_kw = ["Pathogenic", "Likely pathogenic"]
benign_kw = ["Benign", "Likely benign"]
def classify(cs):
    cs = str(cs)
    if any(k in cs for k in patho_kw): return 1
    if any(k in cs for k in benign_kw): return 0
    return -1

snv["label"] = snv["ClinicalSignificance"].apply(classify)
labeled = snv[snv["label"] >= 0].copy()
valid = labeled[
    (labeled["ReferenceAlleleVCF"] != "na") &
    (labeled["AlternateAlleleVCF"] != "na") &
    (labeled["ReferenceAlleleVCF"].str.len() == 1) &
    (labeled["AlternateAlleleVCF"].str.len() == 1)
].copy()
print(f"Valid labeled SNVs: {len(valid):,}")

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

parsed = valid["Name"].apply(parse_hgvs)
valid["tx_id"] = parsed.apply(lambda x: x[0])
valid["cpos"] = parsed.apply(lambda x: x[1])
valid["cref"] = parsed.apply(lambda x: x[2])
valid["calt"] = parsed.apply(lambda x: x[3])

has_all = valid["tx_id"].notna() & valid["cpos"].notna()
print(f"Variants with parseable HGVS: {has_all.sum():,}")

def is_synonymous(name):
    return bool(re.search(r"p\.\w+\d+=", str(name)))

valid["is_synonymous"] = valid["Name"].apply(is_synonymous)

# Stats for all variants with tx_id
with_tx = valid[has_all].copy()
print(f"\n=== All variants with HGVS ===")
print(f"Total: {len(with_tx):,}")
print(f"  Missense: {(~with_tx['is_synonymous']).sum():,} (P={((~with_tx['is_synonymous']) & (with_tx['label']==1)).sum():,}, B={((~with_tx['is_synonymous']) & (with_tx['label']==0)).sum():,})")
print(f"  Synonymous: {with_tx['is_synonymous'].sum():,} (P={(with_tx['is_synonymous'] & (with_tx['label']==1)).sum():,}, B={(with_tx['is_synonymous'] & (with_tx['label']==0)).sum():,})")

# With CDS cache
with_cds = with_tx[with_tx["tx_id"].isin(cds_txids)].copy()
print(f"\n=== Variants with CDS cache ===")
print(f"Total: {len(with_cds):,}")
print(f"  Missense: {(~with_cds['is_synonymous']).sum():,} (P={((~with_cds['is_synonymous']) & (with_cds['label']==1)).sum():,}, B={((~with_cds['is_synonymous']) & (with_cds['label']==0)).sum():,})")
syn_cds = with_cds[with_cds["is_synonymous"]]
print(f"  Synonymous: {len(syn_cds):,} (P={(syn_cds['label']==1).sum():,}, B={(syn_cds['label']==0).sum():,})")

# Find top transcripts NOT in cache that have synonymous pathogenic variants
syn_patho_no_cds = with_tx[with_tx["is_synonymous"] & (with_tx["label"]==1) & ~with_tx["tx_id"].isin(cds_txids)]
tx_counts = syn_patho_no_cds["tx_id"].value_counts()
print(f"\n=== Top transcripts with synonymous pathogenic but NO CDS ===")
print(f"Total unique tx: {len(tx_counts)}")
for tx, cnt in tx_counts.head(30).items():
    print(f"  {tx}: {cnt} syn-patho variants")

# Save top transcripts for CDS fetching
top_new_tx = tx_counts.head(300).index.tolist()
out_path = data_dir / "task2_clinvar" / "top_transcripts_expand.txt"
with open(out_path, "w") as f:
    for tx in top_new_tx:
        f.write(tx + "\n")
print(f"\nSaved {len(top_new_tx)} transcripts to {out_path}")

# Also save all syn-patho tx_ids for reference
all_syn_patho_tx = with_tx[with_tx["is_synonymous"] & (with_tx["label"]==1)]["tx_id"].unique()
print(f"Total unique tx with syn-patho: {len(all_syn_patho_tx)}")
print(f"  Already in CDS cache: {len(set(all_syn_patho_tx) & cds_txids)}")
print(f"  Need CDS: {len(set(all_syn_patho_tx) - cds_txids)}")