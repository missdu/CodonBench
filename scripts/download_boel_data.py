#!/usr/bin/env python3
"""Download Boel et al. 2016 data from available sources."""
import os
import sys

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

data_dir = "./data/regression"
os.makedirs(data_dir, exist_ok=True)

# Try multiple sources for Boel 2016 data
# The key data is in Nature Supplementary Information
# Alternative: use CodonBERT's Fungal_expression.csv as a quick independent test

# First, let's check what CodonBERT already has that we haven't used
import pandas as pd

for fname in ['CoV_Vaccine_Degradation.csv', 'Fungal_expression.csv']:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(f"\n{fname}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Head:\n{df.head(2)}")

# Now try to download Boel 2016 data from alternative sources
import urllib.request

urls_to_try = [
    # Boel 2016 Nature supplementary - try direct Nature URL
    ("boel2016_nature.xlsx", "https://static-content.springer.com/esm/art%3A10.1038%2Fnature16509/MediaObjects/41586_2016_BFnature16509_MOESM834_ESM.xlsx"),
    # Try from a GitHub repo that might have it
    ("boel2016_github.csv", "https://raw.githubusercontent.com/jbkinney/04_codon_usage/master/endogenous_genes.csv"),
]

for fname, url in urls_to_try:
    fpath = os.path.join(data_dir, fname)
    try:
        print(f"\nTrying {url[:80]}...")
        urllib.request.urlretrieve(url, fpath)
        size = os.path.getsize(fpath)
        print(f"  Downloaded: {size} bytes")
        if fname.endswith('.csv'):
            df = pd.read_csv(fpath)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"  Error: {e}")
        if os.path.exists(fpath):
            os.remove(fpath)

# Try openpyxl for xlsx if available
for fname in os.listdir(data_dir):
    if fname.endswith('.xlsx'):
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_excel(fpath)
            print(f"\n{fname}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        except Exception as e:
            print(f"  Error reading {fname}: {e}")

print("\nDone.")