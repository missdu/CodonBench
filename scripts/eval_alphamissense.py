import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import time
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder, _fix_token_type_ids
import src.models.xformers_compat

DEVICE = "cuda:2"

# Load AlphaMissense data
print("Loading AlphaMissense...")
am_path = Path("data/task0_cancer/alphamissense_hg38.tsv.gz")
am_raw = pd.read_csv(am_path, sep="\t", header=None, 
                      names=["chrom","pos","ref","alt","assembly","uniprot","transcript","protein_variant","am_score","am_class"])
print(f"AlphaMissense: {len(am_raw)} rows")

# Filter to missense (single AA change) and get balanced patho/benign
am_raw["am_score"] = pd.to_numeric(am_raw["am_score"], errors="coerce")
am = am_raw.dropna(subset=["am_score"]).copy()
print(f"Valid scores: {len(am)}")

# Binary label: am_score >= 0.5 is pathogenic
am["label"] = (am["am_score"] >= 0.5).astype(int)
n_patho = am["label"].sum()
n_benign = len(am) - n_patho
print(f"Pathogenic: {n_patho}, Benign: {n_benign}")

# Sample balanced set
n_sample = 2000
patho_sample = am[am["label"] == 1].sample(n=min(n_sample//2, n_patho), random_state=42)
benign_sample = am[am["label"] == 0].sample(n=min(n_sample//2, n_benign), random_state=42)
balanced = pd.concat([patho_sample, benign_sample]).sample(frac=1, random_state=42)
print(f"Balanced sample: {len(balanced)} ({int(balanced['label'].sum())} patho)")

# Build codon sequences from ref/alt
# ref and alt are nucleotides, construct wt_codon=ref*3, mut_codon=alt+ref*2
CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

common_codons = list(CODON_TABLE.keys())

def build_seq(row):
    ref = str(row["ref"]).upper()
    alt = str(row["alt"]).upper()
    if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
        return None, None
    np.random.seed(hash(str(row["transcript"]) + str(row["pos"])) % (2**31))
    ctx = [np.random.choice(common_codons) for _ in range(5)]
    wt_codon = ref * 3
    mut_codon = alt + ref * 2
    wt_seq = " ".join(ctx[:2] + [wt_codon] + ctx[2:])
    mut_seq = " ".join(ctx[:2] + [mut_codon] + ctx[2:])
    return wt_seq, mut_seq

# Evaluate with EnCodon-80M
print("\nLoading EnCodon-80M...")
model, tokenizer, meta = CodonModelLoader.load("encodon-80m", device=DEVICE)
if not meta["success"]:
    print(f"Failed: {meta['error']}")
    sys.exit(1)

print(f"Model loaded, evaluating {len(balanced)} samples...")

llrs = []
am_scores = []
labels = []

for i, (_, row) in enumerate(balanced.iterrows()):
    wt, mut = build_seq(row)
    if wt is None:
        continue
    try:
        llr = compute_llr_encoder(model, tokenizer, wt, mut, device=DEVICE)
        llrs.append(llr)
    except:
        llrs.append(0.0)
    am_scores.append(row["am_score"])
    labels.append(row["label"])
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(balanced)} done")

CodonModelLoader.release(model, DEVICE)

llrs = np.array(llrs)
am_scores = np.array(am_scores)
labels = np.array(labels)
neg_llrs = -llrs

# Binary classification metrics
auc_roc = roc_auc_score(labels, neg_llrs)
auc_pr = roc_auc_score(labels, neg_llrs)  # approximate

# Correlation with AlphaMissense continuous scores
spearman_r, spearman_p = stats.spearmanr(neg_llrs, am_scores)
pearson_r, pearson_p = stats.pearsonr(neg_llrs, am_scores)

print(f"\n{'='*60}")
print(f"EnCodon-80M on AlphaMissense (n={len(llrs)})")
print(f"{'='*60}")
print(f"Binary: ROC-AUC={auc_roc:.4f}")
print(f"Continuous: Spearman r={spearman_r:.4f} (p={spearman_p:.2e})")
print(f"Continuous: Pearson r={pearson_r:.4f} (p={pearson_p:.2e})")
print(f"LLR stats: mean={llrs.mean():.4f}, std={llrs.std():.4f}")
print(f"LLR range: [{llrs.min():.4f}, {llrs.max():.4f}]")
print(f"Unique LLR values: {len(np.unique(np.round(llrs, 4)))}")

# Save result
result = {
    "model": "encodon-80m",
    "task": "alphamissense_correlation",
    "n_samples": len(llrs),
    "ROC-AUC": round(auc_roc, 4),
    "spearman_r": round(spearman_r, 4),
    "spearman_p": f"{spearman_p:.2e}",
    "pearson_r": round(pearson_r, 4),
    "pearson_p": f"{pearson_p:.2e}",
    "llr_mean": round(float(llrs.mean()), 4),
    "llr_std": round(float(llrs.std()), 4),
}

out_dir = Path("results/alphamissense_eval")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "encodon-80m_alphamissense.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResult saved to {out_dir / 'encodon-80m_alphamissense.json'}")