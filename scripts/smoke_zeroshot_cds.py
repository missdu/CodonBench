import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import re
import pandas as pd
from pathlib import Path

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder

DEVICE = "cuda:2"
DATA_DIR = Path("./data")

def dna_to_rna(seq):
    return seq.replace("T", "U")

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

def build_cds_sequences(cds_seq, cpos, cref, calt, context_codons=16):
    if cpos < 1 or cpos > len(cds_seq):
        return None, None
    if cds_seq[cpos - 1] != cref:
        return None, None
    wt_seq = cds_seq
    mut_seq = cds_seq[:cpos-1] + calt + cds_seq[cpos:]
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(wt_seq) // 3, codon_idx + context_codons + 1)
    wt_codons = dna_to_codons(wt_seq[start_codon*3:end_codon*3])
    mut_codons = dna_to_codons(mut_seq[start_codon*3:end_codon*3])
    if not wt_codons or not mut_codons:
        return None, None
    return " ".join(wt_codons), " ".join(mut_codons)

print("Loading CDS cache...")
cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
print(f"  {len(cds_cache)} transcripts")

print("Loading ClinVar (500k rows)...")
raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False, nrows=500000)
snv = raw[raw["Type"] == "single nucleotide variant"].copy()

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

parsed = valid["Name"].apply(parse_hgvs)
valid["tx_id"] = parsed.apply(lambda x: x[0])
valid["cpos"] = parsed.apply(lambda x: x[1])
valid["cref"] = parsed.apply(lambda x: x[2])
valid["calt"] = parsed.apply(lambda x: x[3])

has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
variants = valid[has_all].copy()
print(f"  Usable: {len(variants)}")

# Take 5 patho + 5 benign
patho = variants[variants["label"]==1].head(5)
benign = variants[variants["label"]==0].head(5)
sample = pd.concat([patho, benign])
print(f"  Sample: {len(sample)} variants")

# Build sequences
for _, row in sample.iterrows():
    wt_str, mut_str = build_cds_sequences(cds_cache[row["tx_id"]], int(row["cpos"]), row["cref"], row["calt"])
    print(f"\n  {row['Name'][:60]}")
    print(f"  WT:  {wt_str[:80]}...")
    print(f"  MUT: {mut_str[:80]}...")
    # Check they differ at exactly one codon
    wt_c = wt_str.split()
    mut_c = mut_str.split()
    diff_pos = [i for i in range(len(wt_c)) if wt_c[i] != mut_c[i]]
    print(f"  Diff codons at positions: {diff_pos}")

# Test with EnCodon-80M
print("\n=== Testing EnCodon-80M ===")
model, tokenizer, meta = CodonModelLoader.load("encodon-80m", device=DEVICE)
if model is None:
    print(f"FAILED: {meta}")
else:
    for _, row in sample.head(3).iterrows():
        wt_str, mut_str = build_cds_sequences(cds_cache[row["tx_id"]], int(row["cpos"]), row["cref"], row["calt"])
        llr = compute_llr_encoder(model, tokenizer, wt_str, mut_str, device=DEVICE)
        print(f"  {row['Name'][:50]}  LLR={llr:.4f}  label={row['label']}")
    CodonModelLoader.release(model, DEVICE)

print("\nSmoke test PASSED!")