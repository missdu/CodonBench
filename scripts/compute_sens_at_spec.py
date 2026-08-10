"""
Compute sensitivity @ 90% specificity for cLMs on Task 2 and Task 3.
Uses the same pipeline as run_downstream_classification.py.
"""
import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings
from src.data.download import prepare_task2, prepare_task3
import src.models.xformers_compat

DEVICE = "cuda:2"
DATA_DIR = Path("./data")
N_SAMPLES = 5000
N_FOLDS = 5
RANDOM_STATE = 42


def dna_to_rna(seq):
    return seq.replace("T", "U")


def compute_sens_at_spec(y_true, y_prob, spec_target=0.90):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    idx = np.where(specificity >= spec_target)[0]
    if len(idx) == 0:
        return None
    return float(tpr[idx[np.argmax(tpr[idx])]])


def balance_df(df, max_samples, random_state=42):
    n_per_class = max_samples // 2
    df_pos = df[df["label"] == 1].sample(n=min(n_per_class, int(df["label"].sum())), random_state=random_state)
    df_neg = df[df["label"] == 0].sample(n=min(n_per_class, int(len(df) - df["label"].sum())), random_state=random_state)
    return pd.concat([df_pos, df_neg]).sample(frac=1, random_state=random_state)


def run_model_task(model_name, task_name, df, use_rna=False):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, Task: {task_name}")

    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    if model is None:
        print(f"  FAILED: {meta.get('error')}")
        return None

    df_bal = balance_df(df, N_SAMPLES) if len(df) > N_SAMPLES else df

    wt_seqs = df_bal["wt_codon_seq"].astype(str).tolist()
    mut_seqs = df_bal["mut_codon_seq"].astype(str).tolist()
    labels = df_bal["label"].values.astype(int)

    if use_rna:
        wt_seqs = [dna_to_rna(s) for s in wt_seqs]
        mut_seqs = [dna_to_rna(s) for s in mut_seqs]

    print(f"  n={len(labels)}, pos={labels.sum()}, neg={len(labels)-labels.sum()}")

    t0 = time.time()
    wt_embs = extract_embeddings(model, tokenizer, wt_seqs, device=DEVICE, batch_size=16)
    mut_embs = extract_embeddings(model, tokenizer, mut_seqs, device=DEVICE, batch_size=16)
    embed_time = time.time() - t0
    print(f"  Embeddings: {embed_time:.1f}s")

    CodonModelLoader.release(model, DEVICE)

    X = mut_embs - wt_embs

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    fold_sens = []
    all_y_true = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler() if False else None
        if scaler:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        sens = compute_sens_at_spec(y_test, y_prob, spec_target=0.90)

        fold_aucs.append(auc)
        fold_sens.append(sens)
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        sens_str = f"{sens:.4f}" if sens else "N/A"
        print(f"  Fold {fold_idx}: AUC={auc:.4f}, Sens@90%Spec={sens_str}")

    overall_sens = compute_sens_at_spec(np.array(all_y_true), np.array(all_y_prob), spec_target=0.90)

    result = {
        'model': model_name,
        'task': task_name,
        'n_samples': int(len(labels)),
        'auc_mean': float(np.mean(fold_aucs)),
        'auc_std': float(np.std(fold_aucs)),
        'sens_at_90spec_per_fold': fold_sens,
        'sens_at_90spec_mean': float(np.mean([s for s in fold_sens if s is not None])),
        'sens_at_90spec_pooled': overall_sens,
    }
    print(f"  => AUC={result['auc_mean']:.4f}, Sens@90%Spec(pooled)={overall_sens:.4f}")
    return result


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    print("Loading data...")
    df_task2 = prepare_task2(data_dir=DATA_DIR)
    df_task3 = prepare_task3(data_dir=DATA_DIR)
    print(f"Task2: {len(df_task2)} variants, Task3: {len(df_task3)} variants")

    results = []

    models_cfg = [
        ('codonbert', False),
        ('codonbert_hf', True),
        ('encodon-80m', False),
    ]

    for model_name, use_rna in models_cfg:
        for task_name, df in [('task2_missense', df_task2), ('task3_synonymous', df_task3)]:
            r = run_model_task(model_name, task_name, df, use_rna=use_rna)
            if r:
                results.append(r)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    out_path = './results/sensitivity_at_90spec_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
