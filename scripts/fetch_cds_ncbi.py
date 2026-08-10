import requests
import time
import json
import re
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def fetch_cds_ncbi(nm_id, retries=2):
    """Fetch CDS sequence from NCBI GenBank format"""
    for attempt in range(retries):
        try:
            url = f"{BASE}/efetch.fcgi?db=nuccore&id={nm_id}&rettype=gb&retmode=text"
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            text = r.text
            
            cds_match = re.search(r'     CDS             (.+?)(?=\n     [a-z]|\nORIGIN|\nFEATURES)', text, re.DOTALL)
            if not cds_match:
                continue
            
            cds_coords_str = cds_match.group(1).replace('\n', ' ').strip()
            join_match = re.search(r'join\((.+?)\)', cds_coords_str)
            coord_str = join_match.group(1) if join_match else cds_coords_str
            
            coords = []
            for m in re.finditer(r'(\d+)\.\.(\d+)', coord_str):
                coords.append((int(m.group(1)), int(m.group(2))))
            if not coords:
                continue
            
            url2 = f"{BASE}/efetch.fcgi?db=nuccore&id={nm_id}&rettype=fasta&retmode=text"
            r2 = requests.get(url2, timeout=30)
            if r2.status_code != 200:
                continue
            
            lines = r2.text.strip().split('\n')
            full_seq = ''.join(l for l in lines if not l.startswith('>'))
            
            cds_parts = [full_seq[s-1:e] for s, e in coords]
            cds_seq = ''.join(cds_parts)
            
            if cds_seq[:3] == 'ATG' and len(cds_seq) % 3 == 0 and len(cds_seq) > 100:
                return cds_seq
        except:
            if attempt < retries - 1:
                time.sleep(1)
    return None

def main():
    transcript_file = Path("data/task2_clinvar/top_transcripts.txt")
    output_file = Path("data/task2_clinvar/cds_sequences.json")
    
    with open(transcript_file) as f:
        transcripts = [line.strip() for line in f if line.strip()]
    
    print(f"Fetching CDS for {len(transcripts)} transcripts from NCBI...")
    
    cache = {}
    if output_file.exists():
        with open(output_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached entries")
    
    fetched = 0
    failed = 0
    
    for i, tid in enumerate(transcripts):
        if tid in cache and cache[tid] is not None:
            continue
        
        cds = fetch_cds_ncbi(tid)
        cache[tid] = cds
        if cds:
            fetched += 1
        else:
            failed += 1
        
        if (i + 1) % 10 == 0:
            valid = sum(1 for v in cache.values() if v is not None)
            print(f"  {i+1}/{len(transcripts)}: fetched={fetched}, failed={failed}, total_valid={valid}")
        
        if (i + 1) % 50 == 0:
            with open(output_file, "w") as f:
                json.dump(cache, f)
        
        time.sleep(0.4)  # NCBI rate limit: ~3/second without API key
    
    with open(output_file, "w") as f:
        json.dump(cache, f)
    
    valid = sum(1 for v in cache.values() if v is not None)
    lengths = [len(v) for v in cache.values() if v is not None]
    print(f"\nDone! Total: {len(cache)}, Valid CDS: {valid}, Failed: {len(cache)-valid}")
    if lengths:
        import numpy as np
        print(f"CDS length: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")

if __name__ == "__main__":
    main()