"""
onehot_pos 240-fold LOGO-CV
Uses the same unified data splits as step5 (synpath_data/split_logo_cv.npz).
No dependency on data.clinvar_dataset or data.embedding_utils.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import json, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
EVAL_DIR = UNIFIED_DIR / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("./data")

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


def codon_positional_onehot(codons, max_len=33):
    emb = np.zeros(max_len * N_CODONS, dtype=np.float32)
    for i, c in enumerate(codons[:max_len]):
        if c in CODON_TO_IDX:
            emb[i * N_CODONS + CODON_TO_IDX[c]] = 1.0
    return emb


def main():
    print("=== onehot_pos 240-fold LOGO-CV START ===")

    # Load CDS cache
    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))

    # Load ClinVar
    print("Loading ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()

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
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_patho = int((task3_variants["label"] == 1).sum())
    n_benign = int((task3_variants["label"] == 0).sum())
    max_samples = min(n_patho * 2, n_patho + n_benign, 3000)

    df = task3_variants.copy()
    if len(df) > max_samples:
        n_per = min(max_samples // 2, n_patho, n_benign)
        patho = df[df["label"] == 1].sample(n=n_per, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

    # Build onehot_pos features
    records = []
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            continue
        codons = build_cds_sequence(cds_seq, cpos)
        if codons is None:
            continue
        records.append({
            "codons": codons,
            "tx_id": tx,
            "label": row["label"],
        })

    print(f"  {len(records)} synonymous variant samples from "
          f"{len(set(r['tx_id'] for r in records))} transcripts")

    labels = np.array([r["label"] for r in records], dtype=int)
    tx_ids = np.array([r["tx_id"] for r in records])
    features = np.array([codon_positional_onehot(r["codons"]) for r in records])

    # Load the same LOGO-CV splits used by step5
    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    print(f"  Using {len(valid_txs)} LOGO folds (same as step5)")

    # Run LOGO-CV with lbfgs (matching original run_logo_cv.py)
    lr_folds = []
    lr_preds_all, y_all = [], []

    for fi, tx in enumerate(valid_txs):
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        y_tr, y_te = labels[tr], labels[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
            continue
        try:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
            clf.fit(features[tr], y_tr)
            y_prob = clf.predict_proba(features[te])[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            lr_folds.append(auc)
            lr_preds_all.extend(y_prob.tolist())
            y_all.extend(y_te.tolist())
        except Exception:
            continue
        if (fi + 1) % 50 == 0:
            print(f"    LOGO fold {fi+1}/{len(valid_txs)}, "
                  f"mean AUC={np.mean(lr_folds):.4f}±{np.std(lr_folds):.4f}")

    perfold_mean = float(np.mean(lr_folds))
    perfold_sd = float(np.std(lr_folds))
    pooled_auc = roc_auc_score(y_all, lr_preds_all)

    print(f"\n=== RESULTS ===")
    print(f"  Per-fold mean AUC: {perfold_mean:.4f}±{perfold_sd:.4f} (n={len(lr_folds)} folds)")
    print(f"  Pooled AUC: {pooled_auc:.4f}")
    print(f"  Original 52-fold AUC: 0.739")
    print(f"  Standard 5-fold CV AUC: 0.891")

    results = {
        "onehot_pos_synpath_logo_240fold": {
            "lr_perfold_mean": round(perfold_mean, 4),
            "lr_perfold_sd": round(perfold_sd, 4),
            "lr_pooled": round(pooled_auc, 4),
            "n_folds": len(lr_folds),
            "original_52fold_auc": 0.739,
            "standard_5fold_auc": 0.891,
            "drop_from_standard_pp": round((perfold_mean - 0.891) * 100, 1),
        }
    }

    results_file = EVAL_DIR / "onehot_pos_logo_240fold_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {results_file}")

    # Also run with liblinear + no scaler for comparison
    print(f"\n--- Also running liblinear + no scaler ---")
    lr_folds_ll = []
    for fi, tx in enumerate(valid_txs):
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        y_tr, y_te = labels[tr], labels[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
            continue
        try:
            clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
            clf.fit(features[tr], y_tr)
            y_prob = clf.predict_proba(features[te])[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            lr_folds_ll.append(auc)
        except Exception:
            continue

    perfold_mean_ll = float(np.mean(lr_folds_ll))
    print(f"  liblinear per-fold mean: {perfold_mean_ll:.4f} (n={len(lr_folds_ll)} folds)")

    results["onehot_pos_synpath_logo_240fold_liblinear"] = {
        "lr_perfold_mean": round(perfold_mean_ll, 4),
        "n_folds": len(lr_folds_ll),
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("=== onehot_pos 240-fold LOGO-CV DONE ===")


if __name__ == "__main__":
    main()
