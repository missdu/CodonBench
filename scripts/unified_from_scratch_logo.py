"""
Unified From-Scratch LOGO-CV: 2x2 solver x scaler ablation
For all from-scratch conditions (v1-v4, codon & char).

For each (condition, tokenizer, solver, scaler) combo:
  - Run LOGO-CV with LR and MLP
  - Compute per-fold mean AUC and pooled AUC
  - Compute codon-char gap under each configuration
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats as sp_stats

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
EMB_DIR = UNIFIED_DIR / "embeddings" / "from_scratch"
EVAL_DIR = UNIFIED_DIR / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda:1"

CONDITIONS = [
    "codon-v1", "char-v1",
    "codon-v3a", "char-v3a",
    "codon-v3b", "char-v3b",
    "codon-v2", "char-v2",
    "codon-v4", "char-v4",
]

SOLVERS = ["lbfgs", "liblinear"]
SCALERS = ["standard", "none"]

LOG_FILE = EVAL_DIR / "unified_from_scratch_logo.log"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_lr(X_train, y_train, X_test, y_test, solver, use_scaler):
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    lr = LogisticRegression(C=1.0, solver=solver, max_iter=2000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return auc, y_prob


def run_mlp(X_train, y_train, X_test, y_test, input_dim, use_scaler):
    if use_scaler:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s = X_train
        X_test_s = X_test

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)

    mlp = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    mlp.train()
    for epoch in range(200):
        optimizer.zero_grad()
        pred = mlp(X_train_t).squeeze()
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        y_prob = mlp(X_test_t).squeeze().cpu().numpy()
    auc = roc_auc_score(y_test, y_prob)
    return auc, y_prob


def run_logo_cv(emb, y, tx_ids, valid_txs, solver, use_scaler, input_dim):
    lr_folds, mlp_folds = [], []
    lr_preds_all, mlp_preds_all = [], []
    y_all = []

    for fi, tx in enumerate(valid_txs):
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
            continue
        try:
            lr_auc, lr_prob = run_lr(emb[tr], y_tr, emb[te], y_te, solver, use_scaler)
            mlp_auc, mlp_prob = run_mlp(emb[tr], y_tr, emb[te], y_te, input_dim, use_scaler)
            lr_folds.append(lr_auc)
            mlp_folds.append(mlp_auc)
            lr_preds_all.extend(lr_prob.tolist())
            mlp_preds_all.extend(mlp_prob.tolist())
            y_all.extend(y_te.tolist())
        except Exception as e:
            continue
        if (fi + 1) % 50 == 0:
            log(f"    LOGO fold {fi+1}/{len(valid_txs)}, "
                f"LR={np.mean(lr_folds):.3f}, MLP={np.mean(mlp_folds):.3f}")

    if not lr_folds:
        return None

    lr_perfold_mean = float(np.mean(lr_folds))
    mlp_perfold_mean = float(np.mean(mlp_folds))
    gain_perfold = (mlp_perfold_mean - lr_perfold_mean) * 100

    lr_pooled = roc_auc_score(y_all, lr_preds_all)
    mlp_pooled = roc_auc_score(y_all, mlp_preds_all)
    gain_pooled = (mlp_pooled - lr_pooled) * 100

    t_stat, p_value = sp_stats.ttest_rel(mlp_folds, lr_folds)

    return {
        "lr_perfold_mean": round(lr_perfold_mean, 4),
        "lr_perfold_sd": round(float(np.std(lr_folds)), 4),
        "mlp_perfold_mean": round(mlp_perfold_mean, 4),
        "mlp_perfold_sd": round(float(np.std(mlp_folds)), 4),
        "gain_perfold_pp": round(gain_perfold, 1),
        "lr_pooled": round(lr_pooled, 4),
        "mlp_pooled": round(mlp_pooled, 4),
        "gain_pooled_pp": round(gain_pooled, 1),
        "n_folds": len(lr_folds),
        "paired_t": round(float(t_stat), 3),
        "paired_p": round(float(p_value), 4),
    }


def main():
    log("=== Unified From-Scratch LOGO-CV 2x2 START ===")
    log(f"Conditions: {CONDITIONS}")
    log(f"Solvers: {SOLVERS}")
    log(f"Scalers: {SCALERS}")
    log(f"Device: {DEVICE}")

    with open(SYN_DIR / "sequences.json") as f:
        seq_data = json.load(f)
    labels = np.array(seq_data["labels"])
    tx_ids = np.array(seq_data["tx_ids"])

    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    log(f"Loaded {len(labels)} variants, {len(valid_txs)} LOGO folds")

    results_file = EVAL_DIR / "unified_from_scratch_logo_results.json"
    existing = {}
    if results_file.exists():
        existing = json.load(open(results_file))
        log(f"Loaded {len(existing)} existing results")

    for cond in CONDITIONS:
        emb_path = EMB_DIR / f"{cond}_synpath_emb.npy"
        if not emb_path.exists():
            log(f"MISSING {cond}: {emb_path}")
            continue

        emb = np.load(emb_path)
        n = min(len(emb), len(labels))
        emb = emb[:n]
        y = labels[:n]
        tx_arr = tx_ids[:n]
        input_dim = emb.shape[1]

        for solver in SOLVERS:
            for scaler_name in SCALERS:
                use_scaler = (scaler_name == "standard")
                result_key = f"{cond}_{solver}_{scaler_name}"

                if result_key in existing:
                    log(f"  SKIP {result_key} (already done)")
                    continue

                log(f"\n  --- {result_key} ---")
                log(f"  emb={emb.shape}, solver={solver}, scaler={scaler_name}")

                result = run_logo_cv(
                    emb, y, tx_arr, valid_txs,
                    solver, use_scaler, input_dim
                )

                if result is None:
                    log(f"  FAILED: no valid folds")
                    continue

                result["condition"] = cond
                result["solver"] = solver
                result["scaler"] = scaler_name

                existing[result_key] = result
                with open(results_file, "w") as f:
                    json.dump(existing, f, indent=2)

                log(f"  Per-fold: LR={result['lr_perfold_mean']:.3f}±{result['lr_perfold_sd']:.3f} "
                    f"MLP={result['mlp_perfold_mean']:.3f}±{result['mlp_perfold_sd']:.3f} "
                    f"gain={result['gain_perfold_pp']:+.1f}pp")
                log(f"  Pooled:   LR={result['lr_pooled']:.3f} MLP={result['mlp_pooled']:.3f} "
                    f"gain={result['gain_pooled_pp']:+.1f}pp")

        del emb
        torch.cuda.empty_cache()

    log("\n" + "="*60)
    log("=== CODON-CHAR GAP TABLE ===")
    conditions_with_gap = ["v1", "v3a", "v3b", "v4"]
    for solver in SOLVERS:
        for scaler_name in SCALERS:
            log(f"\n  solver={solver}, scaler={scaler_name}:")
            for v in conditions_with_gap:
                codon_key = f"codon-{v}_{solver}_{scaler_name}"
                char_key = f"char-{v}_{solver}_{scaler_name}"
                if codon_key in existing and char_key in existing:
                    c = existing[codon_key]
                    ch = existing[char_key]
                    gap_pf = c['mlp_perfold_mean'] - ch['mlp_perfold_mean']
                    gap_pool = c['mlp_pooled'] - ch['mlp_pooled']
                    log(f"    {v}: codon MLP={c['mlp_perfold_mean']:.3f}(pf)/{c['mlp_pooled']:.3f}(pool) "
                        f"char MLP={ch['mlp_perfold_mean']:.3f}(pf)/{ch['mlp_pooled']:.3f}(pool) "
                        f"gap={gap_pf*100:+.1f}pp(pf)/{gap_pool*100:+.1f}pp(pool)")
                else:
                    log(f"    {v}: missing data")

    log("=== Unified From-Scratch LOGO-CV 2x2 DONE ===")


if __name__ == "__main__":
    main()