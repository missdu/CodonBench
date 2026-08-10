"""CaLM评估脚本 - 使用CaLM官方API提取embedding"""
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/calm_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)


def dna_to_rna_codons(codon_seq):
    return codon_seq.replace("T", "U")


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


def extract_calm_embeddings(sequences, batch_size=32):
    from calm import CaLM
    model = CaLM()
    
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        rna_batch = [dna_to_rna_codons(s) for s in batch]
        embs = model.embed_sequences(rna_batch)
        all_embeddings.append(embs.cpu().numpy() if isinstance(embs, torch.Tensor) else np.array(embs))
    
    return np.vstack(all_embeddings)


def evaluate_calm(task_name, variants_df, cds_cache, max_samples=5000):
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

    sequences = []
    labels = []
    skipped = 0

    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        codon_seq = build_codon_context(cds_seq, cpos)
        if codon_seq is None:
            skipped += 1
            continue
        sequences.append(codon_seq)
        labels.append(row["label"])

    labels = np.array(labels, dtype=int)
    print(f"  {len(sequences)} samples (skipped {skipped}), P={int(labels.sum())} B={int(len(labels)-labels.sum())}")

    if len(sequences) < 100:
        return {"model": "CaLM", "task": task_name, "success": False, "error": "Too few samples"}

    print(f"  Extracting CaLM embeddings...")
    t0 = time.time()
    embeddings = extract_calm_embeddings(sequences, batch_size=32)
    emb_time = time.time() - t0
    print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")

    results = {"model": "CaLM", "task": task_name, "type": "cLM",
               "params_M": 86, "emb_dim": embeddings.shape[1],
               "n_samples": len(sequences), "success": True}

    # LR probing
    print(f"  LR probing (5-fold CV)...")
    clf_lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs_lr = cross_val_score(clf_lr, embeddings, labels, cv=cv, scoring="roc_auc")
    results["lr_auc_mean"] = round(float(aucs_lr.mean()), 4)
    results["lr_auc_std"] = round(float(aucs_lr.std()), 4)
    results["lr_auc_folds"] = [round(float(a), 4) for a in aucs_lr]
    print(f"  LR AUC = {aucs_lr.mean():.4f} ± {aucs_lr.std():.4f}")

    # MLP probing
    print(f"  MLP probing (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            early_stopping=True, random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred = clf_mlp.predict_proba(X_test)[:, 1]
    results["mlp_auc_test"] = round(float(roc_auc_score(y_test, y_pred)), 4)
    print(f"  MLP AUC = {results['mlp_auc_test']:.4f}")

    results["emb_time_s"] = round(emb_time, 1)

    out_file = OUT_DIR / f"calm_{task_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")

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
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_task3_patho = int((task3_variants["label"]==1).sum())
    n_task3_benign = int((task3_variants["label"]==0).sum())
    task3_max = min(n_task3_patho * 2, n_task3_patho + n_task3_benign)

    tasks = [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]

    all_results = []

    for task_name, task_df, max_samples in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating CaLM on {task_name}")
        print(f"{'='*60}")
        try:
            result = evaluate_calm(task_name, task_df, cds_cache, max_samples=max_samples)
            all_results.append(result)
            if result.get("success"):
                print(f"  LR: {result['lr_auc_mean']:.4f}±{result['lr_auc_std']:.4f} | MLP: {result.get('mlp_auc_test', 'N/A')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results.append({"model": "CaLM", "task": task_name, "success": False, "error": str(e)})

    summary_file = OUT_DIR / "calm_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("CaLM evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  CaLM | {r['task']:25s} | LR={r['lr_auc_mean']:.4f}±{r['lr_auc_std']:.4f} | MLP={r.get('mlp_auc_test', 'N/A')}")
        else:
            print(f"  CaLM | {r.get('task','?'):25s} | FAILED: {str(r.get('error','?'))[:80]}")


if __name__ == "__main__":
    main()