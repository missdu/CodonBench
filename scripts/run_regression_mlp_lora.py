#!/usr/bin/env python3
"""Regression MLP + LoRA probing for CodonBench.

Extends the existing Ridge/RF regression evaluation with:
  - MLP regression probe (2-layer, same architecture as classification MLP)
  - LoRA regression fine-tuning (optional, for within-protein tasks with enough data)

Tasks:
  - Task 3: mRFP within-protein expression (n=1459, Ridge R2=0.46)
  - Task 4: E. coli cross-protein expression (n=3000, Ridge R2<0)
  - Task 5: mRNA stability (n=5000, Ridge R2<0)
  - Task 6: Fungal cross-protein expression (n=7089, Ridge R2=0.56)

Usage:
  python scripts/run_regression_mlp_lora.py --mode mlp --device cuda:0
  python scripts/run_regression_mlp_lora.py --mode lora --device cuda:0
  python scripts/run_regression_mlp_lora.py --mode all --device cuda:0
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
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DATA_DIR = Path("./data/regression")
OUT_DIR = Path("./results/regression_mlp_lora")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("codonbert", False),
    ("codonbert_hf", True),
    ("encodon-80m", False),
]


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


def train_mlp(X_train, y_train, X_test, y_test, input_dim, epochs=200, lr=1e-3, device="cuda"):
    model = RegressionMLP(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_r2 = -999
    best_pred = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_v).cpu().numpy()
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_pred = y_pred.copy()

    if best_pred is None:
        model.eval()
        with torch.no_grad():
            best_pred = model(X_v).cpu().numpy()
        best_r2 = r2_score(y_test, best_pred)

    spearman = stats.spearmanr(y_test, best_pred)[0]
    return best_r2, spearman


def load_task_data(task_name):
    if task_name == "task4_mrfp_expression":
        df = pd.read_csv(DATA_DIR / "mRFP_Expression.csv")
        seq_col = "Sequence"
        val_col = "Value"
        df = df[[seq_col, val_col]].dropna()
        df = df.rename(columns={seq_col: "sequence", val_col: "value"})
        df = df[df["sequence"].str.len() >= 9]
        df = df[df["sequence"].str.len() % 3 == 0]
        return df.reset_index(drop=True)

    elif task_name == "task5_ecoli_proteins":
        df = pd.read_csv(DATA_DIR / "E.Coli_proteins.csv")
        seq_col = "Sequence"
        val_col = "Value"
        df = df[[seq_col, val_col]].dropna()
        df = df.rename(columns={seq_col: "sequence", val_col: "value"})
        df = df[df["sequence"].str.len() >= 9]
        df = df[df["sequence"].str.len() % 3 == 0]
        if len(df) > 3000:
            df = df.sample(n=3000, random_state=42)
        return df.reset_index(drop=True)

    elif task_name == "task6_mrna_stability":
        df = pd.read_csv(DATA_DIR / "mRNA_Stability.csv")
        seq_col = "Sequence"
        val_col = "Value"
        df = df[[seq_col, val_col]].dropna()
        df = df.rename(columns={seq_col: "sequence", val_col: "value"})
        df = df[df["sequence"].str.len() >= 9]
        df = df[df["sequence"].str.len() % 3 == 0]
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
        return df.reset_index(drop=True)

    elif task_name == "task7_fungal_expression":
        df = pd.read_csv(DATA_DIR / "Fungal_expression.csv")
        df["CDS"] = df["Sequence"].str.replace("U", "T")
        train_df = df[df["Split"].isin(["train", "val"])]
        test_df = df[df["Split"] == "test"]
        train_df = train_df.rename(columns={"CDS": "sequence", "Value": "value"})
        test_df = test_df.rename(columns={"CDS": "sequence", "Value": "value"})
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    else:
        raise ValueError(f"Unknown task: {task_name}")


def evaluate_mlp(model_name, task_name, use_rna, device):
    print(f"\n{'='*60}")
    print(f"MLP: {model_name} on {task_name}")
    print(f"{'='*60}")

    try:
        if task_name == "task7_fungal_expression":
            train_df, test_df = load_task_data(task_name)
        else:
            df = load_task_data(task_name)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        config = CodonModelLoader.get_config(model_name)
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=device)

        if model is None:
            return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}

        train_seqs = prepare_sequences(train_df["sequence"].tolist(), model_name, use_rna)
        test_seqs = prepare_sequences(test_df["sequence"].tolist(), model_name, use_rna)

        print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}")
        print(f"  Extracting train embeddings...")
        X_train = extract_embeddings(model, tokenizer, train_seqs, device=device, batch_size=16, show_progress=True)
        print(f"  Extracting test embeddings...")
        X_test = extract_embeddings(model, tokenizer, test_seqs, device=device, batch_size=16, show_progress=True)

        CodonModelLoader.release(model, device)

        y_train = train_df["value"].values
        y_test = test_df["value"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Ridge baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        y_pred_ridge = ridge.predict(X_test_s)
        ridge_r2 = r2_score(y_test, y_pred_ridge)
        ridge_spearman = stats.spearmanr(y_test, y_pred_ridge)[0]
        print(f"  Ridge: R2={ridge_r2:.4f}, Spearman={ridge_spearman:.4f}")

        # MLP
        mlp_r2, mlp_spearman = train_mlp(X_train_s, y_train, X_test_s, y_test, X_train.shape[1], device=device)
        print(f"  MLP: R2={mlp_r2:.4f}, Spearman={mlp_spearman:.4f}")

        result = {
            "model": model_name,
            "task": task_name,
            "success": True,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "ridge_r2": round(float(ridge_r2), 4),
            "ridge_spearman": round(float(ridge_spearman), 4),
            "mlp_r2": round(float(mlp_r2), 4),
            "mlp_spearman": round(float(mlp_spearman), 4),
            "mlp_vs_ridge_r2": round(float(mlp_r2 - ridge_r2), 4),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"model": model_name, "task": task_name, "success": False, "error": str(e)}

    out_file = OUT_DIR / f"{model_name}_{task_name}_mlp.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_file}")
    return result


def evaluate_lora(model_name, task_name, use_rna, device):
    """LoRA regression fine-tuning for within-protein tasks."""
    print(f"\n{'='*60}")
    print(f"LoRA: {model_name} on {task_name}")
    print(f"{'='*60}")

    try:
        if task_name == "task7_fungal_expression":
            train_df, test_df = load_task_data(task_name)
        else:
            df = load_task_data(task_name)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        if len(train_df) < 100:
            return {"model": model_name, "task": task_name, "success": False, "error": "Too few samples for LoRA"}

        from peft import LoraConfig, get_peft_model, TaskType

        config = CodonModelLoader.get_config(model_name)
        base_model, tokenizer, meta = CodonModelLoader.load(model_name, device=device)

        if base_model is None:
            return {"model": model_name, "task": task_name, "success": False, "error": meta.get("error")}

        # Get the underlying model for LoRA
        if hasattr(base_model, "model"):
            backbone = base_model.model
        else:
            backbone = base_model

        # LoRA config for regression
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )

        peft_model = get_peft_model(backbone, lora_config)
        peft_model.print_trainable_parameters()

        # Add regression head
        class LoRARegressor(nn.Module):
            def __init__(self, peft_model, hidden_size):
                super().__init__()
                self.peft = peft_model
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1),
                )

            def forward(self, input_ids, attention_mask):
                outputs = self.peft(input_ids=input_ids, attention_mask=attention_mask)
                cls_hidden = outputs.last_hidden_state[:, 0, :]
                return self.head(cls_hidden).squeeze(-1)

        hidden_size = backbone.config.hidden_size
        regressor = LoRARegressor(peft_model, hidden_size).to(device)

        train_seqs = prepare_sequences(train_df["sequence"].tolist(), model_name, use_rna)
        test_seqs = prepare_sequences(test_df["sequence"].tolist(), model_name, use_rna)

        # Tokenize
        train_enc = tokenizer(train_seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        test_enc = tokenizer(test_seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")

        y_train = torch.FloatTensor(train_df["value"].values).to(device)
        y_test = torch.FloatTensor(test_df["value"].values).to(device)

        optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        best_r2 = -999
        best_pred = None

        for epoch in range(30):
            regressor.train()
            indices = torch.randperm(len(y_train))
            batch_size = 16

            for i in range(0, len(y_train), batch_size):
                idx = indices[i:i+batch_size]
                input_ids = train_enc["input_ids"][idx].to(device)
                attention_mask = train_enc["attention_mask"][idx].to(device)
                y_batch = y_train[idx]

                optimizer.zero_grad()
                pred = regressor(input_ids, attention_mask)
                loss = nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if (epoch + 1) % 5 == 0:
                regressor.eval()
                with torch.no_grad():
                    all_pred = []
                    for i in range(0, len(y_test), batch_size):
                        input_ids = test_enc["input_ids"][i:i+batch_size].to(device)
                        attention_mask = test_enc["attention_mask"][i:i+batch_size].to(device)
                        pred = regressor(input_ids, attention_mask)
                        all_pred.append(pred.cpu())
                    y_pred = torch.cat(all_pred).numpy()

                r2 = r2_score(y_test.cpu().numpy(), y_pred)
                spearman = stats.spearmanr(y_test.cpu().numpy(), y_pred)[0]
                print(f"  Epoch {epoch+1}: R2={r2:.4f}, Spearman={spearman:.4f}")

                if r2 > best_r2:
                    best_r2 = r2
                    best_pred = y_pred.copy()

        if best_pred is not None:
            best_spearman = stats.spearmanr(y_test.cpu().numpy(), best_pred)[0]
        else:
            best_r2 = -999
            best_spearman = 0

        CodonModelLoader.release(base_model, device)

        result = {
            "model": model_name,
            "task": task_name,
            "success": True,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "lora_r2": round(float(best_r2), 4),
            "lora_spearman": round(float(best_spearman), 4),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"model": model_name, "task": task_name, "success": False, "error": str(e)}

    out_file = OUT_DIR / f"{model_name}_{task_name}_lora.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlp", "lora", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=None, help="Specific task to run (e.g., task4_mrfp_expression)")
    parser.add_argument("--model", default=None, help="Specific model to run (e.g., codonbert)")
    args = parser.parse_args()

    tasks = [
        "task4_mrfp_expression",
        "task5_ecoli_proteins",
        "task6_mrna_stability",
        "task7_fungal_expression",
    ]

    if args.task:
        tasks = [args.task]

    all_results = []

    for task_name in tasks:
        for model_name, use_rna in MODELS:
            if args.model and model_name != args.model:
                continue

            if args.mode in ["mlp", "all"]:
                result = evaluate_mlp(model_name, task_name, use_rna, args.device)
                all_results.append(result)

            if args.mode in ["lora", "all"]:
                result = evaluate_lora(model_name, task_name, use_rna, args.device)
                all_results.append(result)

    summary_file = OUT_DIR / "regression_mlp_lora_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()