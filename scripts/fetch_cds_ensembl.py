import json
import time
import sys
from pathlib import Path
import requests

def fetch_cds_ensembl(transcript_id, retries=3):
    """Fetch CDS sequence from Ensembl REST API"""
    # Ensembl API: https://rest.ensembl.org/sequence/id/{id}?type=cds
    url = f"https://rest.ensembl.org/sequence/id/{transcript_id}?type=cds"
    headers = {"Content-Type": "text/plain"}
    
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.text.strip()
            elif r.status_code == 400:
                return None  # Invalid ID
            elif r.status_code == 429:
                # Rate limited
                retry_after = int(r.headers.get('Retry-After', 1))
                time.sleep(retry_after)
            else:
                time.sleep(0.5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
    return None

def main():
    transcript_file = Path("data/task2_clinvar/top_transcripts.txt")
    output_file = Path("data/task2_clinvar/cds_sequences.json")
    
    with open(transcript_file) as f:
        transcripts = [line.strip() for line in f if line.strip()]
    
    print(f"Fetching CDS for {len(transcripts)} transcripts...")
    
    # Load existing cache
    cache = {}
    if output_file.exists():
        with open(output_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached sequences")
    
    fetched = 0
    failed = 0
    
    for i, tid in enumerate(transcripts):
        if tid in cache:
            continue
        
        cds = fetch_cds_ensembl(tid)
        if cds and len(cds) > 10:
            cache[tid] = cds
            fetched += 1
        else:
            cache[tid] = None
            failed += 1
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(transcripts)}: fetched={fetched}, failed={failed}, cached={len(cache)}")
            # Save progress periodically
            with open(output_file, "w") as f:
                json.dump(cache, f)
        
        # Ensembl rate limit: ~15 requests/second
        time.sleep(0.1)
    
    # Final save
    with open(output_file, "w") as f:
        json.dump(cache, f)
    
    valid = sum(1 for v in cache.values() if v is not None)
    print(f"\nDone! Total cached: {len(cache)}, valid CDS: {valid}, failed: {len(cache)-valid}")
    
    # Stats
    lengths = [len(v) for v in cache.values() if v is not None]
    if lengths:
        import numpy as np
        print(f"CDS length stats: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")

if __name__ == "__main__":
    main()