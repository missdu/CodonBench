"""
sklearn MLP reproducibility check.
Uses sklearn MLPClassifier with exact same config as run_logo_cv_mlp_v2.py
to confirm that +4.7 pp is reproducible with sklearn MLP.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, json, time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats as sp_stats

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
EMB_DIR_SUPP = Path("./results/supplementary")
EVAL_DIR = UNIFIED_DIR / "eval_results"

def run_lr_sklearn(X_train, y_train, X_test, y_test, use_scaler=True):
    if use_scaler:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_mlp_sklearn(X_train, y_train, X_test, y_test, use_scaler=True):
    if use_scaler:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        solver='adam',
        learning_rate_init=1e-3,
        alpha=1e-4,
    )
    mlp.fit(X_train, y_train)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def main():
    print(f"[{time.strftime('%H:%M:%S')}] === sklearn MLP Reproducibility Check START ===")

    emb = np.load(EMB_DIR_SUPP / "codonbert_task3_synonymous_emb.npy")
    print(f"[{time.strftime('%H:%M:%S')}] Loaded CodonBERT embedding: {emb.shape}")

    with open(SYN_DIR / "sequences.json") as f:
        seq_data = json.load(f)
    labels = np.array(seq_data["labels"])
    tx_ids = np.array(seq_data["tx_ids"])

    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    print(f"[{time.strftime('%H:%M:%S')}] {len(labels)} variants, {len(valid_txs)} LOGO folds")

    n = min(len(emb), len(labels))
    emb = emb[:n]; y = labels[:n]; tx_arr = tx_ids[:n]

    results_file = EVAL_DIR / "sklearn_mlp_repro_results.json"
    results = {}
    if results_file.exists():
        results = json.load(open(results_file))

    for use_scaler in [True, False]:
        tag = "sklearn_StandardScaler" if use_scaler else "sklearn_no_scaler"
        key = f"CodonBERT_SynPath_{tag}"
        if key in results:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP {key}")
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] --- {key} ---")
        t0 = time.time()

        lr_folds, mlp_folds = [], []
        lr_preds_all, mlp_preds_all, y_all = [], [], []

        for fi, tx in enumerate(valid_txs):
            test_mask = tx_arr == tx
            train_mask = ~test_mask
            tr = np.where(train_mask)[0]
            te = np.where(test_mask)[0]
            y_tr, y_te = y[tr], y[te]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
                continue
            try:
                lr_auc, lr_prob = run_lr_sklearn(emb[tr], y_tr, emb[te], y_te, use_scaler)
                mlp_auc, mlp_prob = run_mlp_sklearn(emb[tr], y_tr, emb[te], y_te, use_scaler)
                lr_folds.append(lr_auc)
                mlp_folds.append(mlp_auc)
                lr_preds_all.extend(lr_prob.tolist())
                mlp_preds_all.extend(mlp_prob.tolist())
                y_all.extend(y_te.tolist())
            except:
                continue
            if (fi + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (fi + 1) * (len(valid_txs) - fi - 1)
                print(f"  fold {fi+1}/{len(valid_txs)} | "
                      f"LR={np.mean(lr_folds):.3f} MLP={np.mean(mlp_folds):.3f} | "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

        lr_pf = float(np.mean(lr_folds))
        mlp_pf = float(np.mean(mlp_folds))
        gain_pf = (mlp_pf - lr_pf) * 100
        lr_pool = roc_auc_score(y_all, lr_preds_all)
        mlp_pool = roc_auc_score(y_all, mlp_preds_all)
        gain_pool = (mlp_pool - lr_pool) * 100
        t_stat, p_val = sp_stats.ttest_rel(mlp_folds, lr_folds)

        results[key] = {
            "mlp_impl": "sklearn_MLPClassifier",
            "scaler": tag,
            "lr_perfold_mean": round(lr_pf, 4),
            "lr_perfold_sd": round(float(np.std(lr_folds)), 4),
            "mlp_perfold_mean": round(mlp_pf, 4),
            "mlp_perfold_sd": round(float(np.std(mlp_folds)), 4),
            "gain_perfold_pp": round(gain_pf, 1),
            "lr_pooled": round(lr_pool, 4),
            "mlp_pooled": round(mlp_pool, 4),
            "gain_pooled_pp": round(gain_pool, 1),
            "n_folds": len(lr_folds),
            "paired_t": round(float(t_stat), 3),
            "paired_p": round(float(p_val), 4),
            "sklearn_config": {
                "hidden_layer_sizes": [128, 64],
                "max_iter": 200,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "solver": "adam",
                "learning_rate_init": 1e-3,
                "alpha": 1e-4,
            },
        }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        elapsed = time.time() - t0
        print(f"  DONE in {elapsed:.0f}s | PF: LR={lr_pf:.3f} MLP={mlp_pf:.3f} gain={gain_pf:+.1f}pp | "
              f"Pool: LR={lr_pool:.3f} MLP={mlp_pool:.3f} gain={gain_pool:+.1f}pp | p={p_val:.4f}")

    print(f"\n=== sklearn MLP Reproducibility Check DONE ===")
    print(f"{'Config':<50s} | {'PF gain':>8s} | {'Pool gain':>9s} | {'p':>6s}")
    print("-"*80)
    for k, r in sorted(results.items()):
        print(f"{k:<50s} | {r['gain_perfold_pp']:>+7.1f}pp | {r['gain_pooled_pp']:>+8.1f}pp | {r['paired_p']:>6.4f}")

if __name__ == "__main__":
    main()