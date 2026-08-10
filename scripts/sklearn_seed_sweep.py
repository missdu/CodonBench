"""
Sklearn MLPClassifier seed sweep (20 seeds).
CodonBERT SynPath LOGO-CV only.
Uses sklearn MLPClassifier (M5 config) with varying random_state.

This is the CORRECT seed sweep: tests whether the +4.7 pp gain
from sklearn MLPClassifier (random_state=42) is a lucky seed.
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

SEEDS = list(range(20))

def run_lr(X_train, y_train, X_test, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_mlp_sklearn(X_train, y_train, X_test, y_test, random_state=42):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        solver='adam',
        learning_rate_init=1e-3,
        alpha=1e-4,
    )
    mlp.fit(X_train, y_train)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, mlp_label):
    lr_folds, mlp_folds = [], []
    lr_preds_all, mlp_preds_all, y_all = [], [], []
    t0 = time.time()
    for fi, tx in enumerate(valid_txs):
        test_mask = tx_arr == tx
        train_mask = ~test_mask
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
            continue
        try:
            lr_auc, lr_prob = run_lr(emb[tr], y_tr, emb[te], y_te)
            mlp_auc, mlp_prob = mlp_func(emb[tr], y_tr, emb[te], y_te)
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
    return {
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
    }

def main():
    print(f"[{time.strftime('%H:%M:%S')}] === Sklearn Seed Sweep START ===")
    print(f"[{time.strftime('%H:%M:%S')}] Using sklearn MLPClassifier (M5 config) with varying random_state")

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

    sweep_file = EVAL_DIR / "sklearn_seed_sweep_results.json"
    sweep_results = {}
    if sweep_file.exists():
        sweep_results = json.load(open(sweep_file))

    for seed_val in SEEDS:
        key = f"sklearn_seed_{seed_val}"
        if key in sweep_results:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP random_state={seed_val}")
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] --- random_state={seed_val} ---")

        def mlp_func(X_tr, y_tr, X_te, y_te, rs=seed_val):
            return run_mlp_sklearn(X_tr, y_tr, X_te, y_te, random_state=rs)

        r = run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, f"sklearn_seed_{seed_val}")
        r["random_state"] = seed_val
        r["impl"] = "sklearn_MLPClassifier"
        sweep_results[key] = r
        with open(sweep_file, "w") as f:
            json.dump(sweep_results, f, indent=2)

        print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | Pool gain={r['gain_pooled_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    print(f"\n{'='*80}")
    print(f"=== SKLEARN SEED SWEEP SUMMARY ===")
    gains = [sweep_results[k]["gain_perfold_pp"] for k in sorted(sweep_results.keys())]
    pool_gains = [sweep_results[k]["gain_pooled_pp"] for k in sorted(sweep_results.keys())]
    if gains:
        print(f"  Per-fold gain: mean={np.mean(gains):+.1f}pp, std={np.std(gains):.1f}pp, "
              f"range=[{min(gains):+.1f}, {max(gains):+.1f}]")
        print(f"  Pooled gain:   mean={np.mean(pool_gains):+.1f}pp, std={np.std(pool_gains):.1f}pp, "
              f"range=[{min(pool_gains):+.1f}, {max(pool_gains):+.1f}]")
        if "sklearn_seed_42" in sweep_results:
            r42 = sweep_results["sklearn_seed_42"]
            print(f"  random_state=42: PF gain={r42['gain_perfold_pp']:+.1f}pp (the reported +4.7pp)")
            sorted_gains = sorted(gains)
            rank = sorted_gains.index(r42['gain_perfold_pp']) + 1 if r42['gain_perfold_pp'] in sorted_gains else "?"
            pct = rank / len(sorted_gains) * 100 if isinstance(rank, int) else "?"
            print(f"  Percentile: {pct:.0f}th ({rank}/{len(sorted_gains)})")
        else:
            print(f"  Note: random_state=42 not in sweep (seeds 0-19); M5 anchor uses seed=42")
            m5_file = EVAL_DIR / "m0_m5_ablation_results.json"
            if m5_file.exists():
                m5_data = json.load(open(m5_file))
                if "M5_sklearn_anchor" in m5_data:
                    m5_gain = m5_data["M5_sklearn_anchor"]["gain_perfold_pp"]
                    print(f"  M5 anchor (seed=42): PF gain={m5_gain:+.1f}pp")
                    all_gains = sorted(gains + [m5_gain])
                    rank = all_gains.index(m5_gain) + 1
                    pct = rank / len(all_gains) * 100
                    print(f"  Percentile of seed=42: {pct:.0f}th ({rank}/{len(all_gains)})")

    print(f"\n=== Sklearn Seed Sweep DONE ===")

if __name__ == "__main__":
    main()