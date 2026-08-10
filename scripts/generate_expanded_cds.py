"""
Strategy: Generate large-scale CDS data by combining:
1. Existing CaLM species data (4342 CDS)
2. ClinVar CDS cache (500 CDS)  
3. Synthetic CDS from real codon usage tables (50,000+ CDS)
   - Uses species-specific codon usage frequencies
   - Generates biologically realistic CDS sequences

This ensures both models see the same data distribution,
just tokenized differently.
"""
import json
import os
import numpy as np

OUT_DIR = "./data/expanded_cds"
OUT_FILE = os.path.join(OUT_DIR, "expanded_cds_large.json")

CODON_VOCAB = [
    "TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG",
    "TAT", "TAC", "TAA", "TAG", "TGT", "TGC", "TGA", "TGG",
    "CTT", "CTC", "CTA", "CTG", "CCT", "CCC", "CCA", "CCG",
    "CAT", "CAC", "CAA", "CAG", "CGT", "CGC", "CGA", "CGG",
    "ATT", "ATC", "ATA", "ATG", "ACT", "ACC", "ACA", "ACG",
    "AAT", "AAC", "AAA", "AAG", "AGT", "AGC", "AGA", "AGG",
    "GTT", "GTC", "GTA", "GTG", "GCT", "GCC", "GCA", "GCG",
    "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA", "GGG",
]

STOP_CODONS = {"TAA", "TAG", "TGA"}

# Human codon usage frequencies (from Kazusa codon usage database)
# Format: codon -> frequency per 1000
HUMAN_CODON_FREQ = {
    "TTT": 17.6, "TTC": 20.3, "TTA": 7.7, "TTG": 12.9,
    "TCT": 15.1, "TCC": 17.7, "TCA": 12.2, "TCG": 4.4,
    "TAT": 12.2, "TAC": 15.3, "TAA": 1.0, "TAG": 0.8,
    "TGT": 10.2, "TGC": 12.2, "TGA": 1.2, "TGG": 13.2,
    "CTT": 13.2, "CTC": 19.6, "CTA": 7.2, "CTG": 39.6,
    "CCT": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG": 6.9,
    "CAT": 10.9, "CAC": 15.1, "CAA": 12.3, "CAG": 34.2,
    "CGT": 4.5, "CGC": 10.4, "CGA": 6.2, "CGG": 11.4,
    "ATT": 16.0, "ATC": 20.8, "ATA": 7.5, "ATG": 22.3,
    "ACT": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG": 6.1,
    "AAT": 17.0, "AAC": 18.4, "AAA": 24.4, "AAG": 31.9,
    "AGT": 12.1, "AGC": 19.5, "AGA": 12.2, "AGG": 12.0,
    "GTT": 11.0, "GTC": 14.5, "GTA": 7.1, "GTG": 28.1,
    "GCT": 18.5, "GCC": 27.7, "GCA": 15.8, "GCG": 7.4,
    "GAT": 22.1, "GAC": 25.1, "GAA": 29.0, "GAG": 39.6,
    "GGT": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}

# Build probability distribution for coding codons (exclude stops)
coding_codons = [c for c in CODON_VOCAB if c not in STOP_CODONS]
coding_freqs = np.array([HUMAN_CODON_FREQ[c] for c in coding_codons])
coding_probs = coding_freqs / coding_freqs.sum()

# Amino acid groups (for realistic synonymous substitution)
AA_GROUPS = {
    "Phe": ["TTT", "TTC"], "Leu": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "Ser": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "Tyr": ["TAT", "TAC"], "Cys": ["TGT", "TGC"],
    "Trp": ["TGG"], "Pro": ["CCT", "CCC", "CCA", "CCG"],
    "His": ["CAT", "CAC"], "Gln": ["CAA", "CAG"],
    "Arg": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "Ile": ["ATT", "ATC", "ATA"], "Met": ["ATG"],
    "Thr": ["ACT", "ACC", "ACA", "ACG"], "Asn": ["AAT", "AAC"],
    "Lys": ["AAA", "AAG"], "Val": ["GTT", "GTC", "GTA", "GTG"],
    "Ala": ["GCT", "GCC", "GCA", "GCG"], "Asp": ["GAT", "GAC"],
    "Glu": ["GAA", "GAG"], "Gly": ["GGT", "GGC", "GGA", "GGG"],
}

def generate_realistic_cds(min_len=100, max_len=1500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    n_codons = rng.integers(min_len // 3, max_len // 3)
    codons = ["ATG"]  # Start codon
    for _ in range(n_codons - 2):
        codons.append(rng.choice(coding_codons, p=coding_probs))
    codons.append(rng.choice(list(STOP_CODONS)))  # Stop codon
    return "".join(codons)

print("Loading existing CDS data...")
sequences = []

# 1. CaLM species data
import glob
calm_dir = "<MODEL_PATH>/cLMs/CaLM/data/species"
if os.path.exists(calm_dir):
    for fasta_file in sorted(glob.glob(os.path.join(calm_dir, "*.fasta"))):
        with open(fasta_file) as f:
            seq_lines = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if seq_lines:
                        seq = "".join(seq_lines).upper().replace(" ", "")
                        if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                            sequences.append(seq)
                        seq_lines = []
                else:
                    seq_lines.append(line)
            if seq_lines:
                seq = "".join(seq_lines).upper().replace(" ", "")
                if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                    sequences.append(seq)
    print(f"After CaLM data: {len(sequences)} sequences")

# 2. ClinVar CDS cache
clinvar_cds = "./data/task2_clinvar/cds_sequences.json"
if os.path.exists(clinvar_cds):
    with open(clinvar_cds) as f:
        cds_data = json.load(f)
    for tid, seq in cds_data.items():
        if isinstance(seq, str) and len(seq) >= 30 and seq not in sequences:
            sequences.append(seq)
    print(f"After ClinVar cache: {len(sequences)} sequences")

# 3. Generate synthetic CDS with realistic codon usage
TARGET_TOTAL = 55000
rng = np.random.default_rng(42)
synthetic_count = max(0, TARGET_TOTAL - len(sequences))
print(f"Generating {synthetic_count} synthetic CDS with human codon usage...")

for _ in range(synthetic_count):
    seq = generate_realistic_cds(rng=rng)
    sequences.append(seq)

print(f"Total sequences: {len(sequences)}")

# Save
with open(OUT_FILE, "w") as f:
    json.dump({"sequences": sequences, "source": "calm+clinvar+synthetic_human_codon_usage"}, f)

print(f"Saved to {OUT_FILE}")
print(f"File size: {os.path.getsize(OUT_FILE) / 1e6:.1f} MB")