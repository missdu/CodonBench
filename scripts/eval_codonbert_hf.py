import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import compute_llr_encoder
from src.data.download import prepare_task2
from pathlib import Path
from sklearn.metrics import roc_auc_score

model, tokenizer, meta = CodonModelLoader.load("codonbert_hf", device="cuda:2")
print(f"Model loaded: {meta['success']}, vocab={tokenizer.vocab_size}")

df = prepare_task2(Path("./data"))

def dna_to_rna(seq):
    return seq.replace("T", "U")

llrs = []
labels = []
for i in range(min(500, len(df))):
    row = df.iloc[i]
    wt = dna_to_rna(str(row["wt_codon_seq"]))
    mut = dna_to_rna(str(row["mut_codon_seq"]))
    try:
        llr = compute_llr_encoder(model, tokenizer, wt, mut, device="cuda:2")
        llrs.append(llr)
    except:
        llrs.append(0.0)
    labels.append(row["label"])
    if (i+1) % 100 == 0:
        print(f"  {i+1}/500 done")

llrs = np.array(llrs)
labels = np.array(labels)
auc = roc_auc_score(labels, -llrs)
print(f"\nCodonBERT-HF (RNA format): ROC-AUC={auc:.4f}")
print(f"LLR stats: mean={llrs.mean():.4f}, std={llrs.std():.4f}")

CodonModelLoader.release(model, "cuda:2")