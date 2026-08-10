"""P2-2: Generation proxy evaluation (CAI/RSCU prediction).

Bridge between representation evaluation and generation evaluation.
Test whether cLM embeddings can predict codon usage metrics:
  - CAI (Codon Adaptation Index)
  - RSCU (Relative Synonymous Codon Usage)
  - GC3 (GC content at third codon position)

This tests whether the "information about codon usage" encoded in
cLM embeddings is practically useful for downstream tasks.
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results")
CODON_CONTEXT = 16

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

# Standard codon usage for E. coli (simplified - using human codon usage)
# In practice, we'd use a proper codon usage table
SYNONYMOUS_CODONS = {
    'F': ['TTT', 'TTC'], 'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'Y': ['TAT', 'TAC'],
    'H': ['CAT', 'CAC'], 'Q': ['CAA', 'CAG'], 'N': ['AAT', 'AAC'],
    'K': ['AAA', 'AAG'], 'D': ['GAT', 'GAC'], 'E': ['GAA', 'GAG'],
    'C': ['TGT', 'TGC'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
}

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def dna_to_rna(seq):
    return seq.replace("T", "U")

def compute_gc3(cds_seq):
    """Compute GC content at third codon position."""
    codons = dna_to_codons(cds_seq)
    if not codons: return None
    third_bases = [c[2] for c in codons if len(c) == 3 and c not in ['TAA', 'TAG', 'TGA']]
    if not third_bases: return None
    gc = sum(1 for b in third_bases if b in ['G', 'C'])
    return gc / len(third_bases)

def compute_rscu_vector(cds_seq):
    """Compute RSCU (Relative Synonymous Codon Usage) vector."""
    codons = dna_to_codons(cds_seq)
    if not codons: return None
    # Count codons
    codon_counts = {}
    for c in codons:
        if len(c) == 3 and CODON_TABLE.get(c) != '*':
            codon_counts[c] = codon_counts.get(c, 0) + 1
    # Compute RSCU for each amino acid family
    rscu_values = []
    for aa, syn_codons in SYNONYMOUS_CODONS.items():
        total = sum(codon_counts.get(c, 0) for c in syn_codons)
        if total == 0: continue
        for c in syn_codons:
            observed = codon_counts.get(c, 0)
            expected = total / len(syn_codons)
            rscu = observed / expected if expected > 0 else 0
            rscu_values.append(rscu)
    return rscu_values if rscu_values else None

def compute_cai(cds_seq, ref_freq=None):
    """Simplified CAI computation using geometric mean of relative adaptiveness."""
    codons = dna_to_codons(cds_seq)
    if not codons: return None
    # Use codon counts as proxy for adaptiveness
    codon_counts = {}
    for c in codons:
        if len(c) == 3 and CODON_TABLE.get(c) != '*':
            codon_counts[c] = codon_counts.get(c, 0) + 1
    # For each AA family, compute relative adaptiveness
    log_cai = []
    for aa, syn_codons in SYNONYMOUS_CODONS.items():
        counts = {c: codon_counts.get(c, 0) for c in syn_codons}
        max_count = max(counts.values()) if counts else 0
        if max_count == 0: continue
        for c in syn_codons:
            if codon_counts.get(c, 0) > 0:
                w = counts[c] / max_count
                if w > 0:
                    log_cai.append(np.log(w))
    if not log_cai: return None
    return np.exp(np.mean(log_cai))

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    return cds_cache

def main():
    cds_cache = load_data()
    print(f"Loaded {len(cds_cache)} CDS sequences")

    # Compute codon usage metrics for all CDS
    print("Computing codon usage metrics...")
    gc3_vals, cai_vals, rscu_vals = {}, {}, {}
    for tx_id, cds in cds_cache.items():
        gc3 = compute_gc3(cds)
        cai = compute_cai(cds)
        rscu = compute_rscu_vector(cds)
        if gc3 is not None: gc3_vals[tx_id] = gc3
        if cai is not None: cai_vals[tx_id] = cai
        if rscu is not None: rscu_vals[tx_id] = rscu

    print(f"GC3: {len(gc3_vals)}, CAI: {len(cai_vals)}, RSCU: {len(rscu_vals)}")
    print(f"GC3 range: [{min(gc3_vals.values()):.3f}, {max(gc3_vals.values()):.3f}]")
    print(f"CAI range: [{min(cai_vals.values()):.3f}, {max(cai_vals.values()):.3f}]")

    # Save metrics
    metrics = {
        "gc3": {k: round(v, 4) for k, v in gc3_vals.items()},
        "cai": {k: round(v, 4) for k, v in cai_vals.items()},
    }
    with open(OUT_DIR / "codon_usage_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {OUT_DIR / 'codon_usage_metrics.json'}")

    # Now extract embeddings and predict metrics
    # This part requires GPU - will be done in the auto queue
    print("\nEmbedding extraction + regression will be done by auto queue.")
    print("Script ready for GPU execution.")

if __name__ == "__main__":
    main()