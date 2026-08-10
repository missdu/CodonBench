"""Leave-One-Gene-Out Cross-Validation (LOGO-CV)
解决onehot_pos碾压问题的关键实验

如果LOGO-CV下onehot_pos崩溃而cLM保持 → 论文高光时刻
如果两者都崩溃 → 说明任务有数据泄漏
"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from collections import defaultdict

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/logo_cv")
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


def codon_positional_onehot(codons, max_len=33):
    emb = np.zeros(max_len * N_CODONS, dtype=np.float32)
    for i, c in enumerate(codons[:max_len]):
        if c in CODON_TO_IDX:
            emb[i * N_CODONS + CODON_TO_IDX[c]] = 1.0
    return emb


def codon_onehot_freq(codons):
    emb = np.zeros(N_CODONS, dtype=np.float32)
    for c in codons:
        if c in CODON_TO_IDX:
            emb[CODON_TO_IDX[c]] += 1.0
    total = len(codons)
    if total > 0:
        emb /= total
    return emb


def extract_clm_embeddings(model_name, sequences, device="cuda:2"):
    from src.models.loader import CodonModelLoader
    import src.models.xformers_compat
    from src.eval.evaluation_utils import extract_embeddings

    model, tokenizer, meta = CodonModelLoader.load(model_name, device=device)
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}: {meta.get('error')}")

    emb = extract_embeddings(model, tokenizer, sequences, device=device, batch_size=8)
    CodonModelLoader.release(model, device)
    return emb


def logo_cv(features, labels, gene_ids, min_genes=5, min_per_gene=10):
    unique_genes = gene_ids.unique()
    gene_counts = gene_ids.value_counts()
    valid_genes = gene_counts[gene_counts >= min_per_gene].index

    if len(valid_genes) < min_genes:
        print(f"  WARNING: Only {len(valid_genes)} genes with >= {min_per_gene} samples")
        if len(valid_genes) < 3:
            return None

    gene_aucs = []
    for gene in valid_genes:
        test_mask = gene_ids == gene
        train_mask = ~test_mask

        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            gene_aucs.append({"gene": gene, "auc": auc, "n_train": int(train_mask.sum()),
                             "n_test": int(test_mask.sum()), "p_test": int(y_test.sum())})
        except Exception:
            continue

    if not gene_aucs:
        return None

    aucs = np.array([g["auc"] for g in gene_aucs])
    return {
        "n_genes": len(gene_aucs),
        "auc_mean": round(float(aucs.mean()), 4),
        "auc_std": round(float(aucs.std()), 4),
        "auc_median": round(float(np.median(aucs)), 4),
        "gene_details": gene_aucs[:20],
    }


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = "cuda:2"

    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))

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

    # Use gene (transcript) as the leave-out unit
    # For Task 3 (synonymous) - this is the critical test
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_patho = int((task3_variants["label"]==1).sum())
    n_benign = int((task3_variants["label"]==0).sum())
    max_samples = min(n_patho * 2, n_patho + n_benign, 3000)

    df = task3_variants.copy()
    if len(df) > max_samples:
        n_per = min(max_samples // 2, n_patho, n_benign)
        patho = df[df["label"]==1].sample(n=n_per, random_state=42)
        benign = df[df["label"]==0].sample(n=n_per, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

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
            "codon_seq_str": " ".join(codons),
            "tx_id": tx,
            "label": row["label"],
        })

    print(f"  {len(records)} synonymous variant samples from {len(set(r['tx_id'] for r in records))} genes")

    labels = np.array([r["label"] for r in records], dtype=int)
    gene_ids = np.array([r["tx_id"] for r in records])
    sequences = [r["codon_seq_str"] for r in records]

    all_results = {}

    # === 1. onehot_pos LOGO-CV ===
    print(f"\n{'='*60}")
    print("[1] onehot_pos LOGO-CV")
    print(f"{'='*60}")
    feats_ohp = np.array([codon_positional_onehot(r["codons"]) for r in records])
    res = logo_cv(feats_ohp, labels, gene_ids)
    if res:
        print(f"  LOGO-CV AUC: {res['auc_mean']:.4f}±{res['auc_std']:.4f} (n_genes={res['n_genes']})")
        all_results["onehot_pos"] = res

    # Standard 5-fold CV for comparison
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    aucs_std = cross_val_score(clf, feats_ohp, labels, cv=cv, scoring="roc_auc")
    all_results["onehot_pos"]["standard_cv_auc"] = round(float(aucs_std.mean()), 4)
    print(f"  Standard 5-fold AUC: {aucs_std.mean():.4f}±{aucs_std.std():.4f}")

    # === 2. onehot_freq LOGO-CV ===
    print(f"\n{'='*60}")
    print("[2] onehot_freq LOGO-CV")
    print(f"{'='*60}")
    feats_ohf = np.array([codon_onehot_freq(r["codons"]) for r in records])
    res = logo_cv(feats_ohf, labels, gene_ids)
    if res:
        print(f"  LOGO-CV AUC: {res['auc_mean']:.4f}±{res['auc_std']:.4f} (n_genes={res['n_genes']})")
        all_results["onehot_freq"] = res

    # === 3. cLM LOGO-CV ===
    for clm_name in ["codonbert", "codonbert_hf", "encodon-80m"]:
        print(f"\n{'='*60}")
        print(f"[3] {clm_name} LOGO-CV")
        print(f"{'='*60}")
        try:
            clm_emb = extract_clm_embeddings(clm_name, sequences, device=device)
            res = logo_cv(clm_emb, labels, gene_ids)
            if res:
                print(f"  LOGO-CV AUC: {res['auc_mean']:.4f}±{res['auc_std']:.4f} (n_genes={res['n_genes']})")
                all_results[clm_name] = res

                aucs_std = cross_val_score(clf, clm_emb, labels, cv=cv, scoring="roc_auc")
                all_results[clm_name]["standard_cv_auc"] = round(float(aucs_std.mean()), 4)
                print(f"  Standard 5-fold AUC: {aucs_std.mean():.4f}±{aucs_std.std():.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[clm_name] = {"error": str(e)}

    # Save
    with open(OUT_DIR / "logo_cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print("LOGO-CV Summary (Task 3: synonymous variants)")
    print(f"{'='*60}")
    print(f"{'Method':20s} | {'LOGO-CV AUC':15s} | {'Standard CV':12s} | {'Drop':8s}")
    print("-" * 65)
    for name, res in all_results.items():
        if "auc_mean" in res:
            logo_auc = res["auc_mean"]
            std_auc = res.get("standard_cv_auc", None)
            drop = f"{(logo_auc - std_auc)/std_auc*100:.1f}%" if std_auc else "N/A"
            print(f"{name:20s} | {logo_auc:.4f}±{res['auc_std']:.4f}  | {std_auc or 'N/A':12s} | {drop}")


if __name__ == "__main__":
    main()