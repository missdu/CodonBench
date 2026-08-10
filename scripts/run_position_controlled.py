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
from collections import Counter

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/position_controlled")
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


def codon_onehot_freq(codons):
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


def codon_positional_onehot_shuffled(codons, max_len=33, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    shuffled = codons.copy()
    rng.shuffle(shuffled)
    return codon_positional_onehot(shuffled, max_len)


def position_only_features(codons, cpos, cds_len, max_len=33):
    codon_idx = (cpos - 1) // 3
    pos_norm = codon_idx / max(cds_len // 3, 1)
    pos_bin_N = 1.0 if pos_norm < 0.25 else 0.0
    pos_bin_M = 1.0 if 0.25 <= pos_norm < 0.75 else 0.0
    pos_bin_C = 1.0 if pos_norm >= 0.75 else 0.0
    pos_onehot = np.zeros(max_len, dtype=np.float32)
    local_idx = min(codon_idx, max_len - 1)
    pos_onehot[local_idx] = 1.0
    return np.array([pos_norm, pos_bin_N, pos_bin_M, pos_bin_C], dtype=np.float32)


def codon_freq_plus_position(codons, cpos, cds_len):
    freq = codon_onehot_freq(codons)
    pos = position_only_features(codons, cpos, cds_len)
    return np.concatenate([freq, pos])


def random_position_baseline(codons, cpos, cds_len, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    freq = codon_onehot_freq(codons)
    random_pos = rng.random(4).astype(np.float32)
    return np.concatenate([freq, random_pos])


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


def clm_plus_position(clm_emb, cpos, cds_len):
    pos_norm = cpos / max(cds_len, 1)
    pos_bin_N = 1.0 if pos_norm < 0.25 else 0.0
    pos_bin_M = 1.0 if 0.25 <= pos_norm < 0.75 else 0.0
    pos_bin_C = 1.0 if pos_norm >= 0.75 else 0.0
    pos_feat = np.array([pos_norm, pos_bin_N, pos_bin_M, pos_bin_C], dtype=np.float32)
    return np.concatenate([clm_emb, pos_feat])


def prepare_task_data(task_name, variants_df, cds_cache, max_samples=5000):
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

    records = []
    skipped = 0
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        codons = build_cds_sequence(cds_seq, cpos)
        if codons is None:
            skipped += 1
            continue
        records.append({
            "codons": codons,
            "cpos": cpos,
            "cds_len": len(cds_seq),
            "label": row["label"],
            "codon_seq_str": " ".join(codons),
        })

    print(f"  {len(records)} samples (skipped {skipped})")
    return records


def evaluate_features(features, labels, method_name, task_name):
    labels = np.array(labels, dtype=int)
    result = {"method": method_name, "task": task_name, "feat_dim": features.shape[1]}

    clf_lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs_lr = cross_val_score(clf_lr, features, labels, cv=cv, scoring="roc_auc")
    result["lr_auc_mean"] = round(float(aucs_lr.mean()), 4)
    result["lr_auc_std"] = round(float(aucs_lr.std()), 4)
    result["lr_auc_folds"] = [round(float(a), 4) for a in aucs_lr]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                            early_stopping=True, random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred = clf_mlp.predict_proba(X_test)[:, 1]
    result["mlp_auc_test"] = round(float(roc_auc_score(y_test, y_pred)), 4)

    return result


def position_stratified_eval(features, labels, positions, cds_lengths, method_name, task_name):
    labels = np.array(labels, dtype=int)
    positions = np.array(positions)
    cds_lengths = np.array(cds_lengths)
    pos_ratios = positions / np.maximum(cds_lengths, 1)

    bins = {"N_terminal": (0, 0.25), "middle": (0.25, 0.75), "C_terminal": (0.75, 1.01)}
    results = {"method": method_name, "task": task_name, "stratified": {}}

    for bin_name, (lo, hi) in bins.items():
        mask = (pos_ratios >= lo) & (pos_ratios < hi)
        if mask.sum() < 50:
            continue
        X_bin = features[mask]
        y_bin = labels[mask]
        if len(np.unique(y_bin)) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            aucs = cross_val_score(clf, X_bin, y_bin, cv=cv, scoring="roc_auc")
            results["stratified"][bin_name] = {
                "n": int(mask.sum()),
                "p": int(y_bin.sum()),
                "auc_mean": round(float(aucs.mean()), 4),
                "auc_std": round(float(aucs.std()), 4),
            }
        except Exception:
            pass

    return results


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = "cuda:2"

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
    all_stratified = []

    for task_name, task_df, max_samples in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        records = prepare_task_data(task_name, task_df, cds_cache, max_samples)
        labels = [r["label"] for r in records]

        # === Experiment 1: onehot_pos_shuffled (position destroyed) ===
        print("\n[Exp1] onehot_pos_shuffled (position destroyed)...")
        rng = np.random.default_rng(42)
        feats_shuffled = np.array([
            codon_positional_onehot_shuffled(r["codons"], rng=rng) for r in records
        ])
        res = evaluate_features(feats_shuffled, labels, "onehot_pos_shuffled", task_name)
        all_results.append(res)
        print(f"  LR={res['lr_auc_mean']:.4f}±{res['lr_auc_std']:.4f} MLP={res['mlp_auc_test']:.4f}")

        # === Experiment 2: position_only (no codon identity) ===
        print("\n[Exp2] position_only (no codon identity)...")
        feats_pos = np.array([
            position_only_features(r["codons"], r["cpos"], r["cds_len"]) for r in records
        ])
        res = evaluate_features(feats_pos, labels, "position_only", task_name)
        all_results.append(res)
        print(f"  LR={res['lr_auc_mean']:.4f}±{res['lr_auc_std']:.4f} MLP={res['mlp_auc_test']:.4f}")

        # === Experiment 3: onehot_freq + position ===
        print("\n[Exp3] onehot_freq + position...")
        feats_freq_pos = np.array([
            codon_freq_plus_position(r["codons"], r["cpos"], r["cds_len"]) for r in records
        ])
        res = evaluate_features(feats_freq_pos, labels, "onehot_freq+pos", task_name)
        all_results.append(res)
        print(f"  LR={res['lr_auc_mean']:.4f}±{res['lr_auc_std']:.4f} MLP={res['mlp_auc_test']:.4f}")

        # === Experiment 4: onehot_freq + random_position ===
        print("\n[Exp4] onehot_freq + random_position...")
        rng2 = np.random.default_rng(123)
        feats_freq_rpos = np.array([
            random_position_baseline(r["codons"], r["cpos"], r["cds_len"], rng=rng2) for r in records
        ])
        res = evaluate_features(feats_freq_rpos, labels, "onehot_freq+rand_pos", task_name)
        all_results.append(res)
        print(f"  LR={res['lr_auc_mean']:.4f}±{res['lr_auc_std']:.4f} MLP={res['mlp_auc_test']:.4f}")

        # === Experiment 5: cLM + position ===
        for clm_name in ["codonbert", "codonbert_hf", "encodon-80m"]:
            print(f"\n[Exp5] {clm_name} + position...")
            try:
                sequences = [r["codon_seq_str"] for r in records]
                clm_emb = extract_clm_embeddings(clm_name, sequences, device=device)
                feats_clm_pos = np.array([
                    clm_plus_position(clm_emb[i], records[i]["cpos"], records[i]["cds_len"])
                    for i in range(len(records))
                ])
                res = evaluate_features(feats_clm_pos, labels, f"{clm_name}+pos", task_name)
                all_results.append(res)
                print(f"  LR={res['lr_auc_mean']:.4f}±{res['lr_auc_std']:.4f} MLP={res['mlp_auc_test']:.4f}")

                # Position-stratified evaluation
                positions = [r["cpos"] for r in records]
                cds_lengths = [r["cds_len"] for r in records]
                strat = position_stratified_eval(
                    clm_emb, labels, positions, cds_lengths, clm_name, task_name
                )
                all_stratified.append(strat)

                # Also stratified for onehot_pos
                feats_ohp = np.array([
                    codon_positional_onehot(r["codons"]) for r in records
                ])
                strat_ohp = position_stratified_eval(
                    feats_ohp, labels, positions, cds_lengths, "onehot_pos", task_name
                )
                all_stratified.append(strat_ohp)

            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"method": f"{clm_name}+pos", "task": task_name, "success": False, "error": str(e)})

    # Save results
    with open(OUT_DIR / "position_controlled_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUT_DIR / "position_stratified_results.json", "w") as f:
        json.dump(all_stratified, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Position-controlled ablation complete!")
    print(f"{'='*60}")
    print(f"\n{'Method':30s} | {'Task':25s} | {'LR AUC':12s} | {'MLP AUC':8s} | Dim")
    print("-" * 90)
    for r in all_results:
        if "lr_auc_mean" in r:
            print(f"{r['method']:30s} | {r['task']:25s} | {r['lr_auc_mean']:.4f}±{r['lr_auc_std']:.4f} | {r.get('mlp_auc_test', 'N/A'):8s} | {r.get('feat_dim', '?')}")
        else:
            print(f"{r.get('method','?'):30s} | {r.get('task','?'):25s} | FAILED")

    print(f"\nPosition-stratified results:")
    for s in all_stratified:
        print(f"  {s['method']:20s} | {s['task']:25s} | ", end="")
        for bin_name, bin_res in s.get("stratified", {}).items():
            print(f"{bin_name}: {bin_res['auc_mean']:.3f}(n={bin_res['n']}) ", end="")
        print()


if __name__ == "__main__":
    main()