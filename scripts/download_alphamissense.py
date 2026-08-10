import sys
sys.path.insert(0, ".")
from src.data.download import _try_download
from pathlib import Path

# Download AlphaMissense hg38 (643MB, has genomic coordinates + am_pathogenicity)
url = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
out = Path("data/task0_cancer/alphamissense_hg38.tsv.gz")
ok = _try_download(url, out, timeout=600)
print(f"Download: {ok}")

if ok:
    import pandas as pd
    # Read first few lines to understand format
    df = pd.read_csv(out, sep="\t", comment="#", nrows=20)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
