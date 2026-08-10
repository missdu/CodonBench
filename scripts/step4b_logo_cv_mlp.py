"""
Step 4b: Run LOGO-CV MLP on saved from-scratch embeddings.
Uses the same unified splits from step2 and the saved embeddings from step345.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, time, torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
EMB_DIR = UNIFIED_DIR / "embeddings" / "from_scratch"
EVAL_DIR = UNIFIED_DIR / "eval_results"
DEVICE = "cuda:1"

MODELS = [
    "codon-v1", "char-v1",
    "codon-v3a", "char-v3a",
    "codon-v3b", "char-v3b",
    "codon-v2", "char-v2",
    "codon-v4", "char-v4",
]

LOG_FILE = EVAL_DIR / "logo_cv_mlp.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

def run_mlp(X_train, y_train, X_test, y_test, input_dim):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    mlp = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    mlp.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = mlp(X_train_t).squeeze()
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_test_t).squeeze().cpu().numpy()
    return roc_auc_score(y_test, y_pred)

def main():
    log("=== LOGO-CV MLP for from-scratch models START ===")

    with open(SYN_DIR / "sequences.json") as f:
        seq_data = json.load(f)
    labels = np.array(seq_data["labels"])
    tx_ids = np.array(seq_data["tx_ids"])

    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    log(f"Loaded {len(labels)} variants, {len(valid_txs)} LOGO folds")

    results_file = EVAL_DIR / "from_scratch_logo_cv_mlp_results.json"
    existing = {}
    if results_file.exists():
        existing = json.load(open(results_file))
        log(f"Loaded {len(existing)} existing results")

    for model_name in MODELS:
        if model_name in existing:
            log(f"SKIP {model_name} (already done)")
            continue

        emb_path = EMB_DIR / f"{model_name}_synpath_emb.npy"
        if not emb_path.exists():
            log(f"MISSING {model_name}: {emb_path}")
            continue

        emb = np.load(emb_path)
        n = min(len(emb), len(labels))
        emb = emb[:n]
        y = labels[:n]
        tx_arr = tx_ids[:n]
        input_dim = emb.shape[1]
        log(f"\n{'='*60}\n{model_name} | emb={emb.shape}\n{'='*60}")

        mlp_logo = []
        log(f"  Running LOGO-CV MLP ({len(valid_txs)} folds)...")
        for fi, tx in enumerate(valid_txs):
            test_mask = tx_arr == tx
            train_mask = ~test_mask
            tr = np.where(train_mask)[0]
            te = np.where(test_mask)[0]
            y_tr, y_te = y[tr], y[te]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
                continue
            try:
                auc = run_mlp(emb[tr], y_tr, emb[te], y_te, input_dim)
                mlp_logo.append(auc)
            except:
                continue
            if (fi + 1) % 50 == 0:
                log(f"    LOGO MLP fold {fi+1}/{len(valid_txs)}, mean={np.mean(mlp_logo):.3f}")

        existing[model_name] = {
            "mlp_logo_mean": round(float(np.mean(mlp_logo)), 4) if mlp_logo else None,
            "mlp_logo_n_folds": len(mlp_logo),
            "mlp_logo_folds": [round(float(x), 4) for x in mlp_logo],
        }

        with open(results_file, "w") as f:
            json.dump(existing, f, indent=2)
        log(f"  Saved result for {model_name}: MLP LOGO mean={np.mean(mlp_logo):.3f if mlp_logo else 'N/A'}")

        del emb
        torch.cuda.empty_cache()

    log("\n=== SUMMARY ===")
    lr_results = json.load(open(EVAL_DIR / "from_scratch_lr_single_split_results.json"))
    for k in sorted(existing.keys()):
        mlp_logo = existing[k].get("mlp_logo_mean", "N/A")
        lr_logo = lr_results.get(k, {}).get("lr_logo_mean", "N/A")
        if isinstance(mlp_logo, (int, float)) and isinstance(lr_logo, (int, float)):
            gain = round((mlp_logo - lr_logo) * 100, 1)
            log(f"  {k:15s} | LOGO LR={lr_logo:.3f} MLP={mlp_logo:.3f} gain={gain:+.1f}pp")
        else:
            log(f"  {k:15s} | LOGO LR={lr_logo} MLP={mlp_logo}")

    log("=== LOGO-CV MLP for from-scratch models DONE ===")

if __name__ == "__main__":
    main()