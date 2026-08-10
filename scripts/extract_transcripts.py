import pandas as pd
import re
from pathlib import Path
from collections import Counter

print("Loading ClinVar...")
raw = pd.read_csv("data/task2_clinvar/clinvar_raw.txt.gz", sep="\t", low_memory=False)
snv = raw[raw["Type"] == "single nucleotide variant"].copy()

# Extract transcript IDs from Name column (e.g., NM_003896.4(ST3GAL5):c.206+13C>T)
transcript_pattern = re.compile(r'(NM_\d+\.\d+)')
transcripts = []
for name in snv["Name"].dropna():
    m = transcript_pattern.search(str(name))
    if m:
        transcripts.append(m.group(1))

transcript_counts = Counter(transcripts)
print(f"Total SNVs with transcript: {len(transcripts)}")
print(f"Unique transcripts: {len(transcript_counts)}")
print(f"\nTop 30 transcripts:")
for tid, count in transcript_counts.most_common(30):
    print(f"  {tid}: {count}")

# Save top transcripts for CDS fetching
top_n = 200
top_transcripts = [t for t, c in transcript_counts.most_common(top_n)]
with open("data/task2_clinvar/top_transcripts.txt", "w") as f:
    for t in top_transcripts:
        f.write(t + "\n")
print(f"\nSaved top {top_n} transcripts to data/task2_clinvar/top_transcripts.txt")

# Also extract c. notation for variant positioning
# e.g., c.206+13C>T -> position 206 in CDS
c_pattern = re.compile(r'c\.(\d+)([ACGT])>([ACGT])')
variants_with_pos = 0
for name in snv["Name"].dropna().head(100000):
    m = c_pattern.search(str(name))
    if m:
        variants_with_pos += 1
print(f"\nVariants with c. position (first 100K): {variants_with_pos}")