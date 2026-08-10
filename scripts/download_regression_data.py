import os
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/Sanofi-Public/CodonBERT/master/benchmarks/CodonBERT/data/fine-tune/"
DATA_DIR = Path("./data/regression")
DATA_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "mRFP_Expression.csv",
    "E.Coli_proteins.csv",
    "mRNA_Stability.csv",
    "CoV_Vaccine_Degradation.csv",
    "Fungal_expression.csv",
    "Tc-Riboswitches.csv",
    "MLOS.csv",
]

for fname in files:
    out_path = DATA_DIR / fname
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {fname}: already exists ({size_mb:.1f} MB)")
        continue
    
    print(f"  Downloading {fname}...", end=" ", flush=True)
    try:
        r = requests.get(BASE_URL + fname, timeout=120)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            size_mb = len(r.content) / 1e6
            print(f"OK ({size_mb:.1f} MB)")
        else:
            print(f"FAILED (HTTP {r.status_code})")
    except Exception as e:
        print(f"FAILED ({e})")

print("\nAll downloads complete!")