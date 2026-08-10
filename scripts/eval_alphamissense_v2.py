"""E2: AlphaMissense as a specialized scorer baseline for Task 1 (missense pathogenicity)

Uses pre-computed AlphaMissense scores directly for classification AUC.
No GPU needed - pure CPU evaluation.

Key question: Can a protein-level specialized scorer (AlphaMissense) 
outperform codon-level LMs on missense pathogenicity prediction?
"""
import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/alphamissense_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


def main():
    print("Loading AlphaMissense...")
    am_path = DATA_DIR / "task0_cancer" / "alphamissense_hg38.tsv.gz"
    am_raw = pd.read_csv(am_path, sep="\t", header=None,
                         names=["chrom", "pos", "ref", "alt", "assembly",
                                "uniprot", "transcript", "protein_variant",
                                "am_score", "am_class"])
    am_raw["am_score"] = pd.to_numeric(am_raw["am_score"], errors="coerce")
    am = am_raw.dropna(subset=["am_score"]).copy()
    print(f"AlphaMissense: {len(am)} variants with valid scores")

    am["am_class_str"] = am["am_class"].astype(str)
    print(f"  am_class distribution: {am['am_class_str'].value_counts().to_dict()}")

    print("\nLoading ClinVar...")
    clinvar = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = clinvar[clinvar["Type"] == "single nucleotide variant"].copy()

    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]

    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw):
            return 1
        if any(k in cs for k in benign_kw):
            return 0
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

    has_all = valid["tx_id"].notna() & valid["cpos"].notna()
    variants = valid[has_all].copy()

    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))

    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)

    variants["chrom_clean"] = variants["Chromosome"].astype(str).str.replace("chr", "", regex=False)
    am["chrom_clean"] = am["chrom"].astype(str).str.replace("chr", "", regex=False)

    variants["pos_int"] = pd.to_numeric(variants["Start"], errors="coerce")
    am["pos_int"] = pd.to_numeric(am["pos"], errors="coerce")

    variants["ref_upper"] = variants["ReferenceAlleleVCF"].astype(str).str.upper()
    variants["alt_upper"] = variants["AlternateAlleleVCF"].astype(str).str.upper()
    am["ref_upper"] = am["ref"].astype(str).str.upper()
    am["alt_upper"] = am["alt"].astype(str).str.upper()

    # === Task 1: Missense variants ===
    missense = variants[~variants["is_synonymous"]].copy()
    print(f"\nMissense variants: {len(missense)} ({int(missense['label'].sum())} patho, {int((missense['label']==0).sum())} benign)")


    merged = missense.merge(
        am[["chrom_clean", "pos_int", "ref_upper", "alt_upper", "am_score", "am_class_str"]],
        on=["chrom_clean", "pos_int", "ref_upper", "alt_upper"],
        how="inner"
    )
    print(f"Matched to AlphaMissense: {len(merged)} ({int(merged['label'].sum())} patho, {int((merged['label']==0).sum())} benign)")

    if len(merged) < 100:
        print("Too few matched variants, trying broader match...")
        merged = missense.merge(
            am[["chrom_clean", "pos_int", "ref_upper", "alt_upper", "am_score", "am_class_str"]],
            on=["chrom_clean", "pos_int", "ref_upper", "alt_upper"],
            how="left"
        )
        merged = merged.dropna(subset=["am_score"])
        print(f"Broader match: {len(merged)}")

    if len(merged) < 50:
        print("ERROR: Too few matched variants for reliable evaluation")
        return

    # === Evaluation 1: Full imbalanced set ===
    labels_full = merged["label"].values
    scores_full = merged["am_score"].values

    auc_roc_full = roc_auc_score(labels_full, scores_full)
    auc_pr_full = average_precision_score(labels_full, scores_full)

    print(f"\n{'='*60}")
    print(f"AlphaMissense on Task 1 (Missense) - Full set")
    print(f"{'='*60}")
    print(f"  n = {len(merged)} ({int(labels_full.sum())} patho, {int((labels_full==0).sum())} benign)")
    print(f"  ROC-AUC = {auc_roc_full:.4f}")
    print(f"  PR-AUC  = {auc_pr_full:.4f}")

    # === Evaluation 2: Balanced set (match our 5000-sample protocol) ===
    n_patho = int(labels_full.sum())
    n_benign = int((labels_full == 0).sum())
    n_balanced = min(n_patho, n_benign, 2500)

    if n_patho >= n_balanced and n_benign >= n_balanced:
        patho_df = merged[merged["label"] == 1].sample(n=n_balanced, random_state=42)
        benign_df = merged[merged["label"] == 0].sample(n=n_balanced, random_state=42)
        balanced = pd.concat([patho_df, benign_df]).sample(frac=1, random_state=42)

        labels_bal = balanced["label"].values
        scores_bal = balanced["am_score"].values

        auc_roc_bal = roc_auc_score(labels_bal, scores_bal)
        auc_pr_bal = average_precision_score(labels_bal, scores_bal)

        print(f"\n{'='*60}")
        print(f"AlphaMissense on Task 1 (Missense) - Balanced")
        print(f"{'='*60}")
        print(f"  n = {len(balanced)} ({int(labels_bal.sum())} patho, {int((labels_bal==0).sum())} benign)")
        print(f"  ROC-AUC = {auc_roc_bal:.4f}")
        print(f"  PR-AUC  = {auc_pr_bal:.4f}")
    else:
        auc_roc_bal = None
        auc_pr_bal = None

    # === Evaluation 3: 5-fold CV on balanced set ===
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    if n_patho >= n_balanced and n_benign >= n_balanced:
        X = scores_bal.reshape(-1, 1)
        y = labels_bal
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict_proba(X[test_idx])[:, 1]
            fold_auc = roc_auc_score(y[test_idx], y_pred)
            fold_aucs.append(fold_auc)

        cv_auc = np.mean(fold_aucs)
        cv_std = np.std(fold_aucs)
        print(f"\n{'='*60}")
        print(f"AlphaMissense on Task 1 (Missense) - 5-fold CV (LR on AM score)")
        print(f"{'='*60}")
        print(f"  AUC = {cv_auc:.4f} ± {cv_std:.4f}")
        print(f"  Per-fold: {[round(a, 4) for a in fold_aucs]}")

    # === Evaluation 4: By review status ===
    print(f"\n{'='*60}")
    print(f"AlphaMissense by review status")
    print(f"{'='*60}")

    review_col = "ReviewStatus" if "ReviewStatus" in merged.columns else None
    if review_col:
        for status in sorted(merged[review_col].unique()):
            sub = merged[merged[review_col] == status]
            if len(sub) >= 50 and len(sub["label"].unique()) >= 2:
                sub_auc = roc_auc_score(sub["label"].values, sub["am_score"].values)
                print(f"  {status}: n={len(sub)}, AUC={sub_auc:.4f}")

    # === Evaluation 5: Task 2 (Synonymous) - AlphaMissense should fail ===
    syn_auc = None
    try:
        synonymous = variants[variants["is_synonymous"]].copy()
        syn_merged = synonymous.merge(
            am[["chrom_clean", "pos_int", "ref_upper", "alt_upper", "am_score", "am_class_str"]],
            on=["chrom_clean", "pos_int", "ref_upper", "alt_upper"],
            how="inner"
        )
        print(f"\nSynonymous matched to AlphaMissense: {len(syn_merged)}")
        if len(syn_merged) >= 50 and len(syn_merged["label"].unique()) >= 2:
            n_syn_p = int(syn_merged["label"].sum())
            n_syn_b = int((syn_merged["label"] == 0).sum())
            n_syn_bal = min(n_syn_p, n_syn_b, 2500)
            if n_syn_p >= n_syn_bal and n_syn_b >= n_syn_bal:
                sp = syn_merged[syn_merged["label"] == 1].sample(n=n_syn_bal, random_state=42)
                sb = syn_merged[syn_merged["label"] == 0].sample(n=n_syn_bal, random_state=42)
                syn_bal = pd.concat([sp, sb]).sample(frac=1, random_state=42)
                syn_auc = roc_auc_score(syn_bal["label"].values, syn_bal["am_score"].values)
                print(f"  Task 2 (Synonymous) balanced AUC = {syn_auc:.4f} (n={len(syn_bal)})")
            else:
                syn_auc = roc_auc_score(syn_merged["label"].values, syn_merged["am_score"].values)
                print(f"  Task 2 (Synonymous) AUC = {syn_auc:.4f} (n={len(syn_merged)}, imbalanced)")
        else:
            print("  Too few synonymous variants matched for evaluation")
    except Exception as e:
        print(f"  Synonymous evaluation failed: {e}")

    # === Save results ===
    result = {
        "model": "AlphaMissense",
        "task1_missense": {
            "n_matched": len(merged),
            "n_patho": int(labels_full.sum()),
            "n_benign": int((labels_full == 0).sum()),
            "roc_auc_full": round(auc_roc_full, 4),
            "pr_auc_full": round(auc_pr_full, 4),
        },
        "task1_balanced": {
            "roc_auc": round(auc_roc_bal, 4) if auc_roc_bal is not None else None,
            "pr_auc": round(auc_pr_bal, 4) if auc_pr_bal is not None else None,
        },
        "task1_5fold_cv": {
            "auc_mean": round(cv_auc, 4),
            "auc_std": round(cv_std, 4),
            "fold_aucs": [round(a, 4) for a in fold_aucs],
        } if n_patho >= n_balanced and n_benign >= n_balanced else None,
        "task2_synonymous": {
            "roc_auc": round(syn_auc, 4) if syn_auc is not None else None,
        },
    }

    out_path = OUT_DIR / "alphamissense_classification.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # === Comparison table ===
    print(f"\n{'='*60}")
    print("Comparison: AlphaMissense vs cLMs vs pLMs (Task 1: Missense)")
    print(f"{'='*60}")
    print(f"{'Model':25s} | {'AUC':>8s}")
    print("-" * 40)
    comparisons = [
        ("AlphaMissense (5-fold CV)", round(cv_auc, 4) if n_patho >= n_balanced and n_benign >= n_balanced else auc_roc_full),
        ("ESM-2-650M (5-fold CV)", 0.719),
        ("ESM-1b-650M (5-fold CV)", 0.711),
        ("onehot_pos (5-fold CV)", 0.755),
        ("CodonBERT (5-fold CV)", 0.660),
        ("CodonBERT-HF (5-fold CV)", 0.659),
        ("EnCodon-80M (5-fold CV)", 0.633),
        ("NT-500M (5-fold CV)", 0.570),
    ]
    for name, auc in comparisons:
        print(f"{name:25s} | {auc:.4f}")


if __name__ == "__main__":
    main()