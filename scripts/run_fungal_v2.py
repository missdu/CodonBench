#!/usr/bin/env python3
"""Evaluate cLMs on Fungal expression (independent data source for regression)."""
import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings
import src.models.xformers_compat

DEVICE = "cuda:0"
DATA_DIR = Path("./data/regression")
OUT_DIR = Path("./results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["codonbert", "codonbert_hf", "encodon-80m"]

def main():
    print("Loading Fungal expression data...")
    df = pd.read_csv(DATA_DIR / "Fungal_expression.csv")
    # Convert RNA (U) to DNA (T) for DNA-tokenized models
    df["CDS"] = df["Sequence"].str.replace("U", "T")
    
    train_df = df[df["Split"].isin(["train", "val"])]
    test_df = df[df["Split"] == "test"]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        
        try:
            config = CodonModelLoader.get_config(model_name)
            model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
            
            if model is None:
                print(f"  Failed to load: {meta.get('error')}")
                results[model_name] = {"success": False, "error": meta.get("error")}
                continue
            
            # For CodonBERT-HF, use RNA sequences
            if model_name == "codonbert_hf":
                train_seqs = train_df["Sequence"].tolist()
                test_seqs = test_df["Sequence"].tolist()
            else:
                train_seqs = train_df["CDS"].tolist()
                test_seqs = test_df["CDS"].tolist()
            
            print(f"  Extracting train embeddings (n={len(train_seqs)})...")
            X_train = extract_embeddings(model, tokenizer, train_seqs, device=DEVICE, batch_size=8)
            print(f"  Shape: {X_train.shape}")
            
            print(f"  Extracting test embeddings (n={len(test_seqs)})...")
            X_test = extract_embeddings(model, tokenizer, test_seqs, device=DEVICE, batch_size=8)
            print(f"  Shape: {X_test.shape}")
            
            CodonModelLoader.release(model, DEVICE)
            
            y_train = train_df["Value"].values
            y_test = test_df["Value"].values
            
            # Ridge regression
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_s, y_train)
            y_pred = ridge.predict(X_test_s)
            
            r2 = r2_score(y_test, y_pred)
            spearman = stats.spearmanr(y_test, y_pred)[0]
            
            print(f"  R2: {r2:.4f}, Spearman: {spearman:.4f}")
            results[model_name] = {
                "success": True,
                "r2": float(r2),
                "spearman": float(spearman),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "task": "fungal_expression",
                "source": "Gish et al. 2022 MBE",
                "organism": "fungal",
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {"success": False, "error": str(e)}
    
    out_path = OUT_DIR / "fungal_expression_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()