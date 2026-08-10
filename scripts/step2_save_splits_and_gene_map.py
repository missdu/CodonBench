"""
Step 2: Unified SynPath data loading + split saving + gene/transcript mapping.
Saves all split indices and gene info for reproducibility and gene overlap analysis.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd, time
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/unified_eval/synpath_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 15

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def parse_gene_from_name(name):
    m = re.match(r"NM_\d+\.\d+\((\w+)\):", str(name))
    return m.group(1) if m else None

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def build_char_str(cds, cpos, ctx=CODON_CONTEXT * 3):
    if cpos < 1 or cpos > len(cds): return None
    start = max(0, cpos - 1 - ctx)
    end = min(len(cds), cpos + ctx)
    return cds[start:end]

def main():
    log("=== Step 2: Unified SynPath Data + Splits ===")

    # Load CDS sequences
    log("Loading CDS sequences...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    log(f"  {len(cds_cache)} CDS sequences")

    # Load ClinVar
    log("Loading ClinVar data...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    log(f"  {len(raw)} ClinVar records")

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

    # Parse HGVS
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds_cache.keys()))
    var = v[h].copy()

    # Parse gene symbol from Name AND from GeneSymbol column
    var["gene_symbol_hgvs"] = var["Name"].apply(parse_gene_from_name)
    if "GeneSymbol" in var.columns:
        var["gene_symbol"] = var["GeneSymbol"]
    else:
        var["gene_symbol"] = var["gene_symbol_hgvs"]

    # Filter synonymous
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    syn = var[var["is_syn"]].copy()
    log(f"  {len(syn)} synonymous variants")

    # Balance
    np3 = int((syn["label"] == 1).sum())
    nb3 = int((syn["label"] == 0).sum())
    n = min(np3, nb3)
    balanced = pd.concat([
        syn[syn["label"] == 1].sample(n, random_state=42),
        syn[syn["label"] == 0].sample(n, random_state=42)
    ]).sample(frac=1, random_state=42)
    log(f"  Balanced: {len(balanced)} variants ({int(balanced['label'].sum())} pos, {int(len(balanced)-balanced['label'].sum())} neg)")

    # Build sequences
    codon_seqs, char_seqs, valid_idx = [], [], []
    for i, (idx, row) in enumerate(balanced.iterrows()):
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        cs = build_codon_str(cds, cpos)
        chs = build_char_str(cds, cpos)
        if cs is None or chs is None: continue
        codon_seqs.append(cs)
        char_seqs.append(chs)
        valid_idx.append(i)

    # Filter to valid only
    balanced_valid = balanced.iloc[valid_idx].copy()
    balanced_valid["codon_seq"] = codon_seqs
    balanced_valid["char_seq"] = char_seqs
    balanced_valid = balanced_valid.reset_index(drop=True)

    labels = balanced_valid["label"].values.astype(int)
    tx_ids = balanced_valid["tx_id"].values
    gene_symbols = balanced_valid["gene_symbol"].values
    n = len(balanced_valid)

    log(f"  Final: {n} valid variants, {labels.sum()} pos, {n-labels.sum()} neg")
    log(f"  Unique tx_ids: {len(set(tx_ids))}")
    log(f"  Unique gene_symbols: {len(set(gene_symbols))}")

    # === Save variants ===
    log("Saving variants...")
    balanced_valid.to_parquet(OUT_DIR / "synpath_variants.parquet")

    # === Save sequences ===
    log("Saving sequences...")
    seq_data = {
        "codon_seqs": codon_seqs,
        "char_seqs": char_seqs,
        "labels": labels.tolist(),
        "tx_ids": tx_ids.tolist(),
        "gene_symbols": gene_symbols.tolist(),
    }
    with open(OUT_DIR / "sequences.json", "w") as f:
        json.dump(seq_data, f)

    # === Single 80/20 split ===
    log("Computing single 80/20 split...")
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    np.savez(OUT_DIR / "split_single_8020.npz",
             train_idx=train_idx, test_idx=test_idx,
             train_labels=labels[train_idx], test_labels=labels[test_idx],
             train_tx_ids=tx_ids[train_idx], test_tx_ids=tx_ids[test_idx],
             train_gene_symbols=gene_symbols[train_idx], test_gene_symbols=gene_symbols[test_idx])
    log(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # === 5-fold CV ===
    log("Computing 5-fold CV splits...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_data = {}
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        fold_data[f"fold_{fold}_train"] = train_idx
        fold_data[f"fold_{fold}_test"] = test_idx
        fold_data[f"fold_{fold}_train_labels"] = labels[train_idx]
        fold_data[f"fold_{fold}_test_labels"] = labels[test_idx]
        fold_data[f"fold_{fold}_train_tx_ids"] = tx_ids[train_idx]
        fold_data[f"fold_{fold}_test_tx_ids"] = tx_ids[test_idx]
        fold_data[f"fold_{fold}_train_gene_symbols"] = gene_symbols[train_idx]
        fold_data[f"fold_{fold}_test_gene_symbols"] = gene_symbols[test_idx]
        log(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
    np.savez(OUT_DIR / "split_5fold_cv.npz", **fold_data)

    # === LOGO-CV ===
    log("Computing LOGO-CV splits...")
    unique_txs = sorted(set(tx_ids))
    tx_to_indices = defaultdict(list)
    for i, tx in enumerate(tx_ids):
        tx_to_indices[tx].append(i)

    logo_data = {
        "unique_tx_ids": np.array(unique_txs),
        "n_unique_tx": len(unique_txs),
    }
    for tx in unique_txs:
        logo_data[f"tx_{tx}_indices"] = np.array(tx_to_indices[tx])

    # Also save valid folds info
    valid_folds = []
    for tx in unique_txs:
        test_indices = np.array(tx_to_indices[tx])
        train_indices = np.array([i for i in range(n) if i not in set(tx_to_indices[tx])])
        y_tr = labels[train_indices]
        y_te = labels[test_indices]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(test_indices) < 2:
            continue
        valid_folds.append(tx)
    logo_data["valid_fold_tx_ids"] = np.array(valid_folds)
    logo_data["n_valid_folds"] = len(valid_folds)
    np.savez(OUT_DIR / "split_logo_cv.npz", **logo_data)
    log(f"  {len(unique_txs)} unique tx_ids, {len(valid_folds)} valid folds")

    # === Gene-transcript mapping ===
    log("Saving gene-transcript mapping...")
    gene_to_tx = defaultdict(set)
    tx_to_gene = {}
    for i in range(n):
        gene_to_tx[gene_symbols[i]].add(tx_ids[i])
        tx_to_gene[tx_ids[i]] = gene_symbols[i]

    multi_tx_genes = {g: sorted(list(ts)) for g, ts in gene_to_tx.items() if len(ts) > 1}
    multi_tx_variant_count = sum(1 for i in range(n) if gene_symbols[i] in multi_tx_genes)

    gene_map = {
        "n_variants": n,
        "n_unique_tx_ids": len(set(tx_ids)),
        "n_unique_gene_symbols": len(set(gene_symbols)),
        "n_multi_transcript_genes": len(multi_tx_genes),
        "n_variants_in_multi_tx_genes": multi_tx_variant_count,
        "pct_variants_in_multi_tx_genes": round(100 * multi_tx_variant_count / n, 1),
        "multi_transcript_genes": multi_tx_genes,
        "tx_to_gene": tx_to_gene,
        "gene_to_tx_list": {g: sorted(list(ts)) for g, ts in gene_to_tx.items()},
    }
    with open(OUT_DIR / "gene_transcript_map.json", "w") as f:
        json.dump(gene_map, f, indent=2, default=str)

    log(f"  Multi-transcript genes: {len(multi_tx_genes)} / {len(gene_to_tx)}")
    log(f"  Variants in multi-tx genes: {multi_tx_variant_count} / {n} ({100*multi_tx_variant_count/n:.1f}%)")

    # === Quick gene overlap analysis ===
    log("\n=== Quick Gene Overlap Analysis ===")

    # Single split
    train_genes = set(gene_symbols[train_idx])
    test_genes = set(gene_symbols[test_idx])
    overlap = train_genes & test_genes
    n_test_in_overlap = sum(1 for g in gene_symbols[test_idx] if g in overlap)
    log(f"Single 80/20 split:")
    log(f"  Train genes: {len(train_genes)}, Test genes: {len(test_genes)}")
    log(f"  Overlapping genes: {len(overlap)} ({100*len(overlap)/len(test_genes):.1f}% of test genes)")
    log(f"  Test variants in overlapping genes: {n_test_in_overlap}/{len(test_idx)} ({100*n_test_in_overlap/len(test_idx):.1f}%)")

    train_txs = set(tx_ids[train_idx])
    test_txs = set(tx_ids[test_idx])
    overlap_tx = train_txs & test_txs
    n_test_in_overlap_tx = sum(1 for t in tx_ids[test_idx] if t in overlap_tx)
    log(f"  Overlapping transcripts: {len(overlap_tx)} ({100*len(overlap_tx)/len(test_txs):.1f}% of test txs)")
    log(f"  Test variants in overlapping txs: {n_test_in_overlap_tx}/{len(test_idx)} ({100*n_test_in_overlap_tx/len(test_idx):.1f}%)")

    # 5-fold CV
    log(f"\n5-fold CV:")
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf2.split(np.arange(n), labels)):
        tr_g = set(gene_symbols[tr])
        te_g = set(gene_symbols[te])
        ov = tr_g & te_g
        n_te_ov = sum(1 for g in gene_symbols[te] if g in ov)
        log(f"  Fold {fold}: {len(ov)} overlapping genes ({100*len(ov)/len(te_g):.1f}%), "
            f"{n_te_ov}/{len(te)} test variants in overlap ({100*n_te_ov/len(te):.1f}%)")

    # LOGO-CV
    log(f"\nLOGO-CV:")
    gene_overlap_folds = 0
    for tx in valid_folds:
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        te_g = set(gene_symbols[test_mask])
        tr_g = set(gene_symbols[train_mask])
        if te_g & tr_g:
            gene_overlap_folds += 1
    log(f"  {gene_overlap_folds}/{len(valid_folds)} folds have gene-level overlap "
        f"(same GeneSymbol, different tx_id)")

    log("\n=== Step 2 DONE ===")

if __name__ == "__main__":
    main()