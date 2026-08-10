#!/usr/bin/env python3
"""Find all CDS/fasta data on server for pretraining."""
import os
import glob

search_dirs = [
    "./data",
    "./",
    os.path.expanduser("~"),
]

for d in search_dirs:
    if not os.path.exists(d):
        continue
    for ext in ["*.json", "*.fasta", "*.fa", "*.fna", "*.txt", "*.gz"]:
        for f in glob.glob(os.path.join(d, "**", ext), recursive=True):
            size = os.path.getsize(f)
            if size > 1e4:
                print(f"{size/1e6:.1f}MB  {f}")

# Also check if CodonBERT's pretraining data is available
codonbert_dir = os.path.expanduser("~/CodonBench/cLMs/CodonBERT")
if os.path.exists(codonbert_dir):
    print(f"\nCodonBERT dir: {codonbert_dir}")
    for f in os.listdir(codonbert_dir):
        fp = os.path.join(codonbert_dir, f)
        if os.path.isfile(fp):
            print(f"  {os.path.getsize(fp)/1e6:.1f}MB  {f}")

# Check for any large sequence files
print("\nLarge files (>10MB) in data dir:")
for f in glob.glob("./data/**/*", recursive=True):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        if size > 1e7:
            print(f"  {size/1e6:.1f}MB  {f}")