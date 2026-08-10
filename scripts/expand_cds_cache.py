import sys
import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from Bio import Entrez

DATA_DIR = Path("./data")
Entrez.email = "codonbench@example.com"

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1)
    return None

def fetch_cds_batch(transcripts, existing_cache, max_retries=3):
    new_cache = dict(existing_cache)
    new_count = 0
    failed = []
    
    need = [tx for tx in transcripts if tx not in existing_cache]
    print(f"Need to fetch {len(need)} new transcripts (already have {len(existing_cache)})")
    
    for i, tx in enumerate(need):
        for attempt in range(max_retries):
            try:
                handle = Entrez.efetch(db="nucleotide", id=tx, rettype="fasta", retmode="text")
                from Bio import SeqIO
                record = SeqIO.read(handle, "fasta")
                handle.close()
                seq = str(record.seq).upper()
                if seq.startswith("ATG") and len(seq) % 3 == 0:
                    new_cache[tx] = seq
                    new_count += 1
                else:
                    failed.append(tx)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    failed.append(tx)
                time.sleep(0.5)
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(need)}, new={new_count}, failed={len(failed)}")
            with open(DATA_DIR / "task2_clinvar" / "cds_sequences_expanded.json", "w") as f:
                json.dump(new_cache, f)
        time.sleep(0.4)
    
    return new_cache, new_count, failed

def main():
    print("Loading ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"]
    
    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]
    
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw): return 1
        if any(k in cs for k in benign_kw): return 0
        return -1
    
    snv = snv[snv["ClinicalSignificance"].apply(classify) >= 0].copy()
    
    all_tx = set()
    for name in snv["Name"]:
        tx = parse_hgvs(str(name))
        if tx:
            all_tx.add(tx)
    
    print(f"Total unique transcripts in ClinVar: {len(all_tx)}")
    
    existing_cache = {}
    cache_file = DATA_DIR / "task2_clinvar" / "cds_sequences.json"
    if cache_file.exists():
        existing_cache = json.load(open(cache_file))
        print(f"Existing cache: {len(existing_cache)} transcripts")
    
    tx_list = sorted(all_tx)
    target_count = min(len(tx_list), 1500)
    
    existing_in_list = [tx for tx in tx_list if tx in existing_cache]
    need_list = [tx for tx in tx_list if tx not in existing_cache]
    
    to_fetch = need_list[:target_count - len(existing_in_list)]
    print(f"Will fetch {len(to_fetch)} new transcripts (target: {target_count})")
    
    if to_fetch:
        new_cache, new_count, failed = fetch_cds_batch(to_fetch, existing_cache)
        print(f"Fetched {new_count} new CDS sequences, {len(failed)} failed")
        
        out_file = DATA_DIR / "task2_clinvar" / "cds_sequences_expanded.json"
        with open(out_file, "w") as f:
            json.dump(new_cache, f)
        print(f"Saved expanded cache: {len(new_cache)} transcripts -> {out_file}")
    else:
        print("No new transcripts to fetch")

if __name__ == "__main__":
    main()