import sys
sys.path.insert(0, ".")
import os
import numpy as np
import json
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
from collections import Counter

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/traditional_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16
CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}
N_CODONS = 64

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

def build_cds_sequence(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return codons

def codon_onehot(codons):
    emb = np.zeros(N_CODONS, dtype=np.float32)
    for c in codons:
        if c in CODON_TO_IDX:
            emb[CODON_TO_IDX[c]] += 1.0
    total = len(codons)
    if total > 0:
        emb /= total
    return emb

def codon_positional_onehot(codons, max_len=33):
    emb = np.zeros(max_len * N_CODONS, dtype=np.float32)
    for i, c in enumerate(codons[:max_len]):
        if c in CODON_TO_IDX:
            emb[i * N_CODONS + CODON_TO_IDX[c]] = 1.0
    return emb

def kmer_frequency(seq, k=3, alphabet="ACGT"):
    from itertools import product
    kmers = [''.join(p) for p in product(alphabet, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    freq = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            freq[kmer_to_idx[kmer]] += 1.0
    total = freq.sum()
    if total > 0:
        freq /= total
    return freq

def gc_content(seq):
    gc = sum(1 for c in seq if c in "GCgc")
    return gc / len(seq) if len(seq) > 0 else 0.0

def cai_features(codons):
    from math import log
    if not codons:
        return np.zeros(4, dtype=np.float32)
    gc3 = sum(1 for c in codons if len(c)==3 and c[2] in "GC") / len(codons)
    lengths = np.array([len(c) for c in codons], dtype=np.float32)
    return np.array([gc3, len(codons), lengths.mean(), lengths.std()], dtype=np.float32)

def build_traditional_features(cds_seq, cpos, method="onehot"):
    codons = build_cds_sequence(cds_seq, cpos)
    if codons is None:
        return None
    
    nt_seq = cds_seq[
        max(0, ((cpos-1)//3 - CODON_CONTEXT) * 3):
        min(len(cds_seq), ((cpos-1)//3 + CODON_CONTEXT + 1) * 3)
    ]
    
    if method == "onehot_freq":
        return codon_onehot(codons)
    elif method == "onehot_pos":
        return codon_positional_onehot(codons)
    elif method == "kmer3":
        return kmer_frequency(nt_seq, k=3)
    elif method == "kmer4":
        return kmer_frequency(nt_seq, k=4)
    elif method == "kmer6":
        return kmer_frequency(nt_seq, k=6)
    elif method == "combined":
        oh = codon_onehot(codons)
        k3 = kmer_frequency(nt_seq, k=3)
        gc = np.array([gc_content(nt_seq)], dtype=np.float32)
        cai = cai_features(codons)
        return np.concatenate([oh, k3, gc, cai])
    return None

def evaluate_traditional(method, task_name, variants_df, cds_cache, max_samples=5000):
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)
    
    features = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        feat = build_traditional_features(cds_seq, cpos, method=method)
        if feat is None:
            skipped += 1
            continue
        features.append(feat)
        labels.append(row["label"])
    
    features = np.array(features)
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(features)} samples (skipped {skipped}), P={int(labels.sum())} B={int(len(labels)-labels.sum())}, feat_dim={features.shape[1]}")
    
    if len(features) < 100:
        return {"model": method, "task": task_name, "success": False, "error": f"Too few: {len(features)}"}
    
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    t0 = time.time()
    aucs = cross_val_score(clf, features, labels, cv=cv, scoring="roc_auc")
    cv_time = time.time() - t0
    
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    
    t_stat, p_val = stats.ttest_1samp(aucs, 0.5)
    neg_log10_p = -np.log10(p_val) if p_val > 0 else float("inf")
    
    result = {
        "model": method, "task": task_name,
        "model_type": "traditional", "feat_dim": features.shape[1],
        "success": True, "n_samples": len(features),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC_mean": round(float(mean_auc), 4),
        "ROC-AUC_std": round(float(std_auc), 4),
        "ROC-AUC_folds": [round(float(a), 4) for a in aucs],
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "cv_time_s": round(cv_time, 1),
        "context_codons": CODON_CONTEXT,
        "input_type": "traditional_features",
    }
    
    result_file = OUT_DIR / f"{method}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result: ROC-AUC={mean_auc:.4f}+/-{std_auc:.4f}, -log10(p)={result['neg_log10_p']}")
    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")
    
    print("Loading FULL ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    print(f"  SNVs: {len(snv):,}")
    
    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw): return 1
        if any(k in cs for k in benign_kw): return 0
        return -1
    
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    labeled = snv[snv["label"] >= 0].copy()
    valid = labeled[
        (labeled["ReferenceAlleleVCF"] != "na") &
        (labeled["AlternateAlleleVCF"] != "na") &
        (labeled["ReferenceAlleleVCF"].str.len() == 1) &
        (labeled["AlternateAlleleVCF"].str.len() == 1)
    ].copy()
    
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    
    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()
    
    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))
    
    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()
    print(f"  Task2 (missense): {len(task2_variants):,}")
    print(f"  Task3 (synonymous): {len(task3_variants):,}")
    
    traditional_methods = ["onehot_freq", "onehot_pos", "kmer3", "kmer4", "kmer6", "combined"]
    
    n_task3_patho = int((task3_variants["label"]==1).sum())
    n_task3_benign = int((task3_variants["label"]==0).sum())
    task3_max = min(n_task3_patho * 2, n_task3_patho + n_task3_benign)
    
    tasks = [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]
    
    all_results = []
    for method in traditional_methods:
        for task_name, task_df, max_samples in tasks:
            if len(task_df) < 100:
                continue
            print(f"\n{'='*60}")
            print(f"Evaluating {method} on {task_name} (max={max_samples})")
            print(f"{'='*60}")
            try:
                result = evaluate_traditional(
                    method, task_name, task_df, cds_cache,
                    max_samples=max_samples
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"model": method, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "traditional_baselines_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Traditional baseline evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:15s} | {r['task']:25s} | AUC={r['ROC-AUC_mean']:.4f}+/-{r['ROC-AUC_std']:.4f} | n={r['n_samples']} | dim={r['feat_dim']}")
        else:
            print(f"  {r['model']:15s} | {r.get('task','?'):25s} | FAILED: {str(r.get('error','unknown'))[:80]}")

if __name__ == "__main__":
    main()