#!/usr/bin/env python3
"""Fair regression evaluation: 5-fold CV with validation-based early stopping.

Protocol (matching Supplementary Table S21):
  - 5-fold CV (KFold, shuffle=True, random_state=42)
  - Within each fold: 80% outer-train, 20% test
  - Outer-train split 80/20 into inner-train / validation
  - MLP trained for max 200 epochs; epoch with best validation R2 selected
  - Ridge alpha selected by validation R2 over [0.01, 0.1, 1.0, 10.0, 100.0]
  - Evaluate on test fold; report R2 and gain (MLP - Ridge) in pp

Models: codonbert, codonbert_hf, encodon-80m
Tasks:  task4_mrfp_expression (n=1459), task7_fungal_expression (n=7089)

Usage:
  python scripts/run_regression_fair_cv.py --device cuda:0
  python scripts/run_regression_fair_cv.py --device cuda:0 --model codonbert_hf --task task4_mrfp_expression
"""
import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ[k] = ""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DATA_DIR = Path("./data/regression")
OUT_DIR = Path("./results/regression_fair_cv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("codonbert", False),
    ("codonbert_hf", True),
    ("encodon-80m", False),
]

TASKS = [
    "task4_mrfp_expression",
    "task7_fungal_expression",
]

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
MAX_EPOCHS = 200
N_FOLDS = 5
SEED = 42


def codonize(seq):
    return " ".join([seq[i:i+3] for i in range(0, len(seq) - 2, 3)])


def prepare_sequences(sequences, model_name, use_rna):
    processed = []
    for seq in sequences:
        seq = seq.upper().strip()
        if use_rna:
            seq = seq.replace("T", "U")
            processed.append(codonize(seq))
        elif model_name == "codonbert":
            dna = seq.replace("U", "T")
            processed.append(codonize(dna))
        else:
            dna = seq.replace("U", "T")
            processed.append(dna)
    return processed


def load_task_data(task_name):
    if task_name == "task4_mrfp_expression":
        df = pd.read_csv(DATA_DIR / "mRFP_Expression.csv")
        seq_col, val_col = "Sequence", "Value"
        df = df[[seq_col, val_col]].dropna()
        df = df.rename(columns={seq_col: "sequence", val_col: "value"})
        df = df[df["sequence"].str.len() >= 9]
        df = df[df["sequence"].str.len() % 3 == 0]
        return df.reset_index(drop=True)

    elif task_name == "task7_fungal_expression":
        df = pd.read_csv(DATA_DIR / "Fungal_expression.csv")
        df["CDS"] = df["Sequence"].str.replace("U", "T")
        train_df = df[df["Split"].isin(["train", "val"])].copy()
        test_df = df[df["Split"] == "test"].copy()
        train_df = train_df.rename(columns={"CDS": "sequence", "Value": "value"})
        test_df = test_df.rename(columns={"CDS": "sequence", "Value": "value"})
        combined = pd.concat([train_df, test_df], ignore_index=True)
        return combined.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown task: {task_name}")


class RegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fair_mlp_fold(X_inner_train, y_inner_train, X_val, y_val, X_test, y_test,
                  input_dim, device="cuda"):
    mlp = RegressionMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.MSELoss()

    X_it = torch.FloatTensor(X_inner_train).to(device)
    y_it = torch.FloatTensor(y_inner_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    X_te = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_it, y_it)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_val_r2 = -999
    best_test_r2 = None
    best_epoch = -1

    for epoch in range(MAX_EPOCHS):
        mlp.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = mlp(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        mlp.eval()
        with torch.no_grad():
            val_pred = mlp(X_v).cpu().numpy()
            val_r2 = r2_score(y_val, val_pred)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            with torch.no_grad():
                test_pred = mlp(X_te).cpu().numpy()
                best_test_r2 = r2_score(y_test, test_pred)

    if best_test_r2 is None:
        mlp.eval()
        with torch.no_grad():
            test_pred = mlp(X_te).cpu().numpy()
            best_test_r2 = r2_score(y_test, test_pred)
        best_epoch = MAX_EPOCHS - 1

    return best_test_r2, best_epoch


def fair_ridge_fold(X_inner_train, y_inner_train, X_val, y_val, X_test, y_test):
    best_val_r2 = -999
    best_alpha = 1.0
    best_test_r2 = None

    for alpha in RIDGE_ALPHAS:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_inner_train, y_inner_train)
        val_pred = ridge.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha
            test_pred = ridge.predict(X_test)
            best_test_r2 = r2_score(y_test, test_pred)

    return best_test_r2, best_alpha


def run_fair_evaluation(model_name, task_name, use_rna, device):
    print(f"\n{'='*60}")
    print(f"Fair CV: {model_name} on {task_name}")
    print(f"{'='*60}")

    try:
        df = load_task_data(task_name)
        print(f"  Data loaded: {len(df)} samples")

        config = CodonModelLoader.get_config(model_name)
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=device)

        if model is None:
            err = meta.get("error", "unknown")
            print(f"  FAILED to load {model_name}: {err}")
            return {"model": model_name, "task": task_name, "success": False, "error": err}

        sequences = prepare_sequences(df["sequence"].tolist(), model_name, use_rna)
        values = df["value"].values.astype(float)

        print(f"  Extracting embeddings ({len(sequences)} seqs)...")
        t0 = time.time()
        embeddings = extract_embeddings(model, tokenizer, sequences, device=device,
                                       batch_size=16, show_progress=True)
        embed_time = time.time() - t0
        print(f"  Embeddings: shape={embeddings.shape}, time={embed_time:.1f}s")

        CodonModelLoader.release(model, device)

        X = embeddings
        y = values

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            X_inner_train, X_val, y_inner_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=SEED
            )

            scaler = StandardScaler()
            X_inner_train_s = scaler.fit_transform(X_inner_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)

            ridge_r2, ridge_alpha = fair_ridge_fold(
                X_inner_train_s, y_inner_train, X_val_s, y_val, X_test_s, y_test
            )

            mlp_r2, mlp_epoch = fair_mlp_fold(
                X_inner_train_s, y_inner_train, X_val_s, y_val, X_test_s, y_test,
                input_dim=X.shape[1], device=device
            )

            gain_pp = (mlp_r2 - ridge_r2) * 100

            fold_results.append({
                "fold": fold_idx,
                "ridge_r2": round(float(ridge_r2), 6),
                "ridge_alpha": round(float(ridge_alpha), 4),
                "mlp_r2": round(float(mlp_r2), 6),
                "mlp_best_epoch": mlp_epoch,
                "gain_pp": round(float(gain_pp), 2),
            })
            print(f"  Fold {fold_idx}: Ridge R2={ridge_r2:.4f} (alpha={ridge_alpha}), "
                  f"MLP R2={mlp_r2:.4f} (epoch={mlp_epoch}), Gain={gain_pp:+.2f}pp")

        mean_ridge = np.mean([f["ridge_r2"] for f in fold_results])
        std_ridge = np.std([f["ridge_r2"] for f in fold_results])
        mean_mlp = np.mean([f["mlp_r2"] for f in fold_results])
        std_mlp = np.std([f["mlp_r2"] for f in fold_results])
        mean_gain = np.mean([f["gain_pp"] for f in fold_results])

        result = {
            "model": model_name,
            "task": task_name,
            "success": True,
            "n_samples": int(len(X)),
            "n_folds": N_FOLDS,
            "embedding_dim": int(X.shape[1]),
            "embed_time_s": round(embed_time, 1),
            "folds": fold_results,
            "mean_ridge_r2": round(float(mean_ridge), 6),
            "std_ridge_r2": round(float(std_ridge), 6),
            "mean_mlp_r2": round(float(mean_mlp), 6),
            "std_mlp_r2": round(float(std_mlp), 6),
            "mean_gain_pp": round(float(mean_gain), 2),
        }

        print(f"\n  SUMMARY: Ridge R2={mean_ridge:.4f}±{std_ridge:.4f}, "
              f"MLP R2={mean_mlp:.4f}±{std_mlp:.4f}, "
              f"Gain={mean_gain:+.2f}pp")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"model": model_name, "task": task_name, "success": False, "error": str(e)}

    out_file = OUT_DIR / f"{model_name}_{task_name}_fair_cv.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None, help="Specific model (codonbert, codonbert_hf, encodon-80m)")
    parser.add_argument("--task", default=None, help="Specific task (task4_mrfp_expression, task7_fungal_expression)")
    args = parser.parse_args()

    all_results = []

    for model_name, use_rna in MODELS:
        if args.model and model_name != args.model:
            continue
        for task_name in TASKS:
            if args.task and task_name != args.task:
                continue

            out_file = OUT_DIR / f"{model_name}_{task_name}_fair_cv.json"
            if out_file.exists():
                print(f"  SKIP {model_name} {task_name} (already exists: {out_file})")
                with open(out_file) as f:
                    existing = json.load(f)
                all_results.append(existing)
                continue

            result = run_fair_evaluation(model_name, task_name, use_rna, args.device)
            all_results.append(result)

    summary_file = OUT_DIR / "regression_fair_cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fair CV Summary ({len(all_results)} model-task pairs)")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:15s} | {r['task']:30s} | "
                  f"Ridge={r['mean_ridge_r2']:.4f}±{r['std_ridge_r2']:.4f} | "
                  f"MLP={r['mean_mlp_r2']:.4f}±{r['std_mlp_r2']:.4f} | "
                  f"Gain={r['mean_gain_pp']:+.2f}pp")
        else:
            print(f"  {r['model']:15s} | {r['task']:30s} | FAILED: {r.get('error','')[:60]}")

    print(f"\nSaved to {summary_file}")


if __name__ == "__main__":
    main()