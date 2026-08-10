"""
Download human CDS sequences via NCBI Entrez (Biopython).
Downloads in batches of 10000, targets 50000 sequences.
"""
import json
import os
import sys
import time

from Bio import Entrez, SeqIO

Entrez.email = "codonbench@example.com"

OUT_DIR = "./data/expanded_cds"
OUT_FILE = os.path.join(OUT_DIR, "human_cds_entrez.json")

if os.path.exists(OUT_FILE):
    with open(OUT_FILE) as f:
        existing = json.load(f)
    if len(existing) >= 10000:
        print(f"Already have {len(existing)} CDS. Done.")
        sys.exit(0)

cds_sequences = {}

# Step 1: Search for human coding sequences
print("Searching NCBI for human CDS sequences...")
handle = Entrez.esearch(
    db="nucleotide",
    term='Homo sapiens[Organism] AND biomol_mRNA[Properties] AND refseq[Filter]',
    retmax=50000,
    usehistory="y",
)
results = Entrez.read(handle)
handle.close()

count = int(results["Count"])
print(f"Found {count} sequences")

webenv = results["WebEnv"]
query_key = results["QueryKey"]

# Step 2: Fetch sequences in batches
batch_size = 500
total_saved = 0

for start in range(0, min(count, 50000), batch_size):
    print(f"Fetching batch starting at {start}...")
    try:
        handle = Entrez.efetch(
            db="nucleotide",
            rettype="fasta_cds_na",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=query_key,
        )
        
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        
        for rec in records:
            seq = str(rec.seq).upper()
            # Filter: must be valid CDS (ATGC only, length multiple of 3, >= 30nt)
            if (len(seq) >= 30 and len(seq) % 3 == 0 and 
                all(c in "ATGC" for c in seq) and
                seq[:3] in ["ATG"] and  # Must start with ATG
                seq[-3:] in ["TAA", "TAG", "TGA"]):  # Must end with stop codon
                cds_sequences[rec.id] = seq
                total_saved += 1
        
        print(f"  Batch done. Total valid CDS: {len(cds_sequences)}")
        
        # Save checkpoint every 5 batches
        if (start // batch_size) % 5 == 0:
            with open(OUT_FILE, "w") as f:
                json.dump(cds_sequences, f)
            print(f"  Checkpoint saved: {len(cds_sequences)} sequences")
        
        time.sleep(0.5)  # Rate limit
        
    except Exception as e:
        print(f"  Error at batch {start}: {e}")
        time.sleep(5)
        continue
    
    if len(cds_sequences) >= 50000:
        break

with open(OUT_FILE, "w") as f:
    json.dump(cds_sequences, f)

print(f"\nDone! Saved {len(cds_sequences)} CDS sequences to {OUT_FILE}")
