"""
Analyze gene overlap under three splitting strategies:
1. Single 80/20 split (train_test_split, random_state=42, stratify=labels)
2. 5-fold CV (StratifiedKFold, n_splits=5, shuffle=True, random_state=42)
3. LOGO-CV (leave-one-transcript-out)

For each split, compute:
- How many unique genes (GeneSymbol) in train and test
- How many genes overlap between train and test
- What fraction of test variants belong to genes also in train
- Same for transcripts (tx_id)
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np
import json
import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = "./data"

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def parse_gene_from_hgvs(name):
    m = re.match(r"NM_\d+\.\d+\((\w+)\):", str(name))
    return m.group(1) if m else None

def load_synpath():
    cds_file = f"{DATA_DIR}/task2_clinvar/cds_sequences.json"
    with open(cds_file) as f:
        cds_cache = json.load(f)
    
    clinvar_file = f"{DATA_DIR}/task2_clinvar/clinvar_raw.txt.gz"
    raw = pd.read_csv(clinvar_file, sep="\t", low_memory=False)
    
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    pk, bk = ["Pathogenic", "Likely pathogenic"], ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]
    
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds_cache.keys()))
    var = v[h].copy()
    
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    syn = var[var["is_syn"]].copy()
    
    np3 = int((syn["label"] == 1).sum())
    nb3 = int((syn["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)
    n = min(max3 // 2, np3, nb3)
    balanced = pd.concat([
        syn[syn["label"] == 1].sample(n, random_state=42),
        syn[syn["label"] == 0].sample(n, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    # Extract gene symbol from Name column
    balanced["gene_symbol"] = balanced["Name"].apply(parse_gene_from_hgvs)
    # Also try GeneSymbol column if available
    if "GeneSymbol" in balanced.columns:
        balanced["gene_symbol"] = balanced["GeneSymbol"]
    
    return balanced

def analyze_split(train_idx, test_idx, gene_ids, tx_ids, labels, split_name):
    train_genes = set(gene_ids[train_idx])
    test_genes = set(gene_ids[test_idx])
    train_txs = set(tx_ids[train_idx])
    test_txs = set(tx_ids[test_idx])
    
    overlap_genes = train_genes & test_genes
    overlap_txs = train_txs & test_txs
    
    n_train = len(train_idx)
    n_test = len(test_idx)
    
    # How many test variants belong to overlapping genes?
    test_gene_arr = gene_ids[test_idx]
    test_tx_arr = tx_ids[test_idx]
    n_test_in_overlap_gene = sum(1 for g in test_gene_arr if g in overlap_genes)
    n_test_in_overlap_tx = sum(1 for t in test_tx_arr if t in overlap_txs)
    
    print(f"\n{'='*60}")
    print(f"  {split_name}")
    print(f"{'='*60}")
    print(f"  Train: {n_train} variants, {len(train_genes)} genes, {len(train_txs)} transcripts")
    print(f"  Test:  {n_test} variants, {len(test_genes)} genes, {len(test_txs)} transcripts")
    print(f"  --- Gene-level overlap ---")
    print(f"  Overlapping genes: {len(overlap_genes)} / {len(test_genes)} test genes ({100*len(overlap_genes)/max(len(test_genes),1):.1f}%)")
    print(f"  Test variants in overlapping genes: {n_test_in_overlap_gene} / {n_test} ({100*n_test_in_overlap_gene/max(n_test,1):.1f}%)")
    print(f"  --- Transcript-level overlap ---")
    print(f"  Overlapping transcripts: {len(overlap_txs)} / {len(test_txs)} test transcripts ({100*len(overlap_txs)/max(len(test_txs),1):.1f}%)")
    print(f"  Test variants in overlapping transcripts: {n_test_in_overlap_tx} / {n_test} ({100*n_test_in_overlap_tx/max(n_test,1):.1f}%)")
    
    return {
        "split": split_name,
        "n_train": n_train, "n_test": n_test,
        "n_train_genes": len(train_genes), "n_test_genes": len(test_genes),
        "n_train_txs": len(train_txs), "n_test_txs": len(test_txs),
        "n_overlap_genes": len(overlap_genes),
        "pct_overlap_genes": 100*len(overlap_genes)/max(len(test_genes),1),
        "n_test_in_overlap_gene": n_test_in_overlap_gene,
        "pct_test_in_overlap_gene": 100*n_test_in_overlap_gene/max(n_test,1),
        "n_overlap_txs": len(overlap_txs),
        "pct_overlap_txs": 100*len(overlap_txs)/max(len(test_txs),1),
        "n_test_in_overlap_tx": n_test_in_overlap_tx,
        "pct_test_in_overlap_tx": 100*n_test_in_overlap_tx/max(n_test,1),
    }

def main():
    print("Loading SynPath data...")
    df = load_synpath()
    print(f"  Total: {len(df)} variants")
    print(f"  Unique gene symbols: {df['gene_symbol'].nunique()}")
    print(f"  Unique transcripts: {df['tx_id'].nunique()}")
    
    # Check multi-transcript genes
    gene_to_tx = defaultdict(set)
    for _, row in df.iterrows():
        gene_to_tx[row['gene_symbol']].add(row['tx_id'])
    multi_tx = {g: ts for g, ts in gene_to_tx.items() if len(ts) > 1}
    print(f"  Genes with multiple transcripts: {len(multi_tx)} / {len(gene_to_tx)}")
    if multi_tx:
        multi_tx_variants = sum(len(df[df['gene_symbol'] == g]) for g in multi_tx)
        print(f"  Variants in multi-transcript genes: {multi_tx_variants} / {len(df)} ({100*multi_tx_variants/len(df):.1f}%)")
        for g, ts in sorted(multi_tx.items(), key=lambda x: -len(x[1]))[:10]:
            n = len(df[df['gene_symbol'] == g])
            print(f"    {g}: {len(ts)} transcripts, {n} variants, tx_ids={sorted(ts)[:3]}...")
    
    gene_ids = df["gene_symbol"].values
    tx_ids = df["tx_id"].values
    labels = df["label"].values
    n = len(df)
    indices = np.arange(n)
    
    results = []
    
    # 1. Single 80/20 split
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    r = analyze_split(train_idx, test_idx, gene_ids, tx_ids, labels, "Single 80/20 split (random_state=42)")
    results.append(r)
    
    # 2. 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        r = analyze_split(train_idx, test_idx, gene_ids, tx_ids, labels, f"5-fold CV fold {fold+1}")
        fold_results.append(r)
    
    # Aggregate 5-fold CV
    avg_overlap_genes = np.mean([r["pct_overlap_genes"] for r in fold_results])
    avg_overlap_txs = np.mean([r["pct_overlap_txs"] for r in fold_results])
    avg_test_in_overlap_gene = np.mean([r["pct_test_in_overlap_gene"] for r in fold_results])
    avg_test_in_overlap_tx = np.mean([r["pct_test_in_overlap_tx"] for r in fold_results])
    print(f"\n  5-fold CV AVERAGES:")
    print(f"  Avg gene overlap: {avg_overlap_genes:.1f}% of test genes overlap with train")
    print(f"  Avg transcript overlap: {avg_overlap_txs:.1f}% of test transcripts overlap with train")
    print(f"  Avg test variants in overlapping genes: {avg_test_in_overlap_gene:.1f}%")
    print(f"  Avg test variants in overlapping transcripts: {avg_test_in_overlap_tx:.1f}%")
    results.extend(fold_results)
    
    # 3. LOGO-CV (leave-one-transcript-out)
    unique_txs = sorted(set(tx_ids))
    print(f"\n  LOGO-CV: {len(unique_txs)} unique transcripts")
    
    logo_overlap_genes_list = []
    logo_overlap_txs_list = []
    n_valid_folds = 0
    for tx in unique_txs:
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        y_tr = labels[train_idx]
        y_te = labels[test_idx]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        if len(test_idx) < 2:
            continue
        
        n_valid_folds += 1
        train_genes = set(gene_ids[train_idx])
        test_genes = set(gene_ids[test_idx])
        overlap_genes = train_genes & test_genes
        
        # In LOGO-CV, transcript overlap should be 0 by design
        train_txs = set(tx_ids[train_idx])
        test_txs = set(tx_ids[test_idx])
        overlap_txs = train_txs & test_txs
        
        n_test_in_overlap_gene = sum(1 for g in gene_ids[test_idx] if g in overlap_genes)
        
        logo_overlap_genes_list.append(len(overlap_genes) / max(len(test_genes), 1) * 100)
        logo_overlap_txs_list.append(len(overlap_txs) / max(len(test_txs), 1) * 100)
    
    print(f"  Valid folds: {n_valid_folds}")
    print(f"  Avg gene overlap: {np.mean(logo_overlap_genes_list):.1f}% of test genes also in train")
    print(f"  Avg transcript overlap: {np.mean(logo_overlap_txs_list):.1f}% (should be 0%)")
    print(f"  Folds with gene overlap: {sum(1 for x in logo_overlap_genes_list if x > 0)} / {n_valid_folds}")
    
    # Count how many folds have gene overlap (same gene, different transcript)
    gene_overlap_folds = 0
    gene_overlap_examples = []
    for tx in unique_txs:
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        
        test_genes_for_tx = set(gene_ids[test_idx])
        train_genes_for_tx = set(gene_ids[train_idx])
        overlap = test_genes_for_tx & train_genes_for_tx
        
        if overlap:
            gene_overlap_folds += 1
            if len(gene_overlap_examples) < 10:
                gene_overlap_examples.append({
                    "held_out_tx": tx,
                    "held_out_gene": list(test_genes_for_tx)[0] if len(test_genes_for_tx) == 1 else list(test_genes_for_tx),
                    "overlapping_genes": list(overlap),
                    "n_test": len(test_idx),
                    "n_train": len(train_idx),
                })
    
    print(f"\n  LOGO-CV GENE-LEVEL LEAKAGE ANALYSIS:")
    print(f"  Folds where held-out gene also appears in train: {gene_overlap_folds} / {n_valid_folds}")
    print(f"  This means: same GeneSymbol, different transcript (NM_xxx)")
    if gene_overlap_examples:
        print(f"\n  Examples of gene-level leakage in LOGO-CV:")
        for ex in gene_overlap_examples:
            print(f"    Held-out tx: {ex['held_out_tx']}, gene: {ex['held_out_gene']}, "
                  f"overlap genes: {ex['overlapping_genes']}, n_test={ex['n_test']}")
    
    # Save results
    out = {
        "dataset": "SynPath",
        "n_variants": int(len(df)),
        "n_unique_genes": int(df['gene_symbol'].nunique()),
        "n_unique_transcripts": int(df['tx_id'].nunique()),
        "n_multi_transcript_genes": len(multi_tx),
        "single_split": results[0],
        "five_fold_cv_avg": {
            "avg_pct_overlap_genes": float(avg_overlap_genes),
            "avg_pct_overlap_txs": float(avg_overlap_txs),
            "avg_pct_test_in_overlap_gene": float(avg_test_in_overlap_gene),
            "avg_pct_test_in_overlap_tx": float(avg_test_in_overlap_tx),
        },
        "logo_cv": {
            "n_valid_folds": n_valid_folds,
            "avg_pct_gene_overlap": float(np.mean(logo_overlap_genes_list)),
            "avg_pct_tx_overlap": float(np.mean(logo_overlap_txs_list)),
            "n_folds_with_gene_overlap": gene_overlap_folds,
            "gene_overlap_examples": gene_overlap_examples,
        }
    }
    
    with open("results/gene_overlap_analysis.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to results/gene_overlap_analysis.json")

if __name__ == "__main__":
    main()