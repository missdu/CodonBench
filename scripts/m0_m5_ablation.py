"""
M0→M5 single-variable ablation + seed sweep (20 seeds).
CodonBERT SynPath LOGO-CV only.
All PyTorch-based (M0-M4), sklearn-based (M5).

M0: baseline (no ES, no seed, 100 ep, no wd)
M1: +fixed seed 42
M2: +early_stopping (KEY STEP)
M3: +weight_decay 1e-4
M4: +epochs 200
M5: sklearn MLPClassifier (anchor, should ≈ +4.7 pp)

Seed sweep: M2 config with 20 different seeds.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, json, time, torch, torch.nn as nn
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
DEVICE = "cuda:1"

SEEDS = list(range(20))

class EarlyStoppingMLP(nn.Module):
    def __init__(self, input_dim, hidden=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def run_lr(X_train, y_train, X_test, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_mlp_pytorch(X_train, y_train, X_test, y_test, input_dim,
                    use_early_stopping=False, seed=None, max_epochs=100,
                    weight_decay=0.0, val_fraction=0.1, patience=10):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)

    set_seed(seed)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    model = EarlyStoppingMLP(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    if use_early_stopping:
        n_val = max(1, int(len(X_train) * val_fraction))
        n_tr_actual = len(X_train) - n_val
        if n_val < 2 or n_tr_actual < 10:
            use_early_stopping = False
        else:
            perm = torch.randperm(len(X_train))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            X_es_tr = X_tr_t[tr_idx]
            y_es_tr = y_tr_t[tr_idx]
            X_es_val = X_tr_t[val_idx]
            y_es_val = y_tr_t[val_idx]

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    model.train()
    for ep in range(max_epochs):
        if use_early_stopping:
            opt.zero_grad()
            pred = model(X_es_tr).squeeze()
            loss_fn(pred, y_es_tr).backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_es_val).squeeze()
                val_loss = loss_fn(val_pred, y_es_val).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            opt.zero_grad()
            pred = model(X_tr_t).squeeze()
            loss_fn(pred, y_tr_t).backward()
            opt.step()

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_prob = model(X_te_t).squeeze().cpu().numpy()
    return roc_auc_score(y_test, y_prob), y_prob

def run_mlp_sklearn(X_train, y_train, X_test, y_test):
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

MLP_CONFIGS = [
    {"name": "M0_baseline", "use_es": False, "seed": None, "epochs": 100, "wd": 0.0},
    {"name": "M1_plus_seed", "use_es": False, "seed": 42, "epochs": 100, "wd": 0.0},
    {"name": "M2_plus_early_stop", "use_es": True, "seed": 42, "epochs": 100, "wd": 0.0},
    {"name": "M3_plus_weight_decay", "use_es": True, "seed": 42, "epochs": 100, "wd": 1e-4},
    {"name": "M4_plus_epochs200", "use_es": True, "seed": 42, "epochs": 200, "wd": 1e-4},
]

def run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, mlp_label, input_dim=None):
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
            if input_dim is not None:
                mlp_auc, mlp_prob = mlp_func(emb[tr], y_tr, emb[te], y_te, input_dim)
            else:
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
    print(f"[{time.strftime('%H:%M:%S')}] === M0→M5 Ablation + Seed Sweep START ===")

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
    input_dim = emb.shape[1]

    results_file = EVAL_DIR / "m0_m5_ablation_results.json"
    results = {}
    if results_file.exists():
        results = json.load(open(results_file))

    # M0→M4: PyTorch single-variable ablation
    for cfg in MLP_CONFIGS:
        key = cfg["name"]
        if key in results:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP {key}")
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] --- {key} ---")
        print(f"  early_stop={cfg['use_es']}, seed={cfg['seed']}, epochs={cfg['epochs']}, wd={cfg['wd']}")

        def mlp_func(X_tr, y_tr, X_te, y_te, inp_dim=input_dim,
                     use_es=cfg["use_es"], seed=cfg["seed"],
                     epochs=cfg["epochs"], wd=cfg["wd"]):
            return run_mlp_pytorch(X_tr, y_tr, X_te, y_te, inp_dim,
                                   use_early_stopping=use_es, seed=seed,
                                   max_epochs=epochs, weight_decay=wd)

        r = run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, key, input_dim)
        r["config"] = cfg
        results[key] = r
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | Pool gain={r['gain_pooled_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    # M5: sklearn anchor
    key = "M5_sklearn_anchor"
    if key not in results:
        print(f"\n[{time.strftime('%H:%M:%S')}] --- {key} (sklearn MLPClassifier) ---")
        def sklearn_func(X_tr, y_tr, X_te, y_te):
            return run_mlp_sklearn(X_tr, y_tr, X_te, y_te)
        r = run_logo_cv(emb, y, tx_arr, valid_txs, sklearn_func, key)
        r["config"] = {"name": "M5_sklearn", "impl": "sklearn_MLPClassifier",
                       "early_stopping": True, "seed": 42, "epochs": 200, "weight_decay": 1e-4}
        results[key] = r
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | Pool gain={r['gain_pooled_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    # Seed sweep: M2 config with 20 different seeds
    print(f"\n[{time.strftime('%H:%M:%S')}] === Seed Sweep (M2 config, 20 seeds) ===")
    sweep_file = EVAL_DIR / "seed_sweep_results.json"
    sweep_results = {}
    if sweep_file.exists():
        sweep_results = json.load(open(sweep_file))

    for seed_val in SEEDS:
        key = f"seed_{seed_val}"
        if key in sweep_results:
            print(f"[{time.strftime('%H:%M:%S')}] SKIP seed={seed_val}")
            continue

        print(f"\n[{time.strftime('%H:%M:%S')}] --- seed={seed_val} ---")

        def mlp_func_seed(X_tr, y_tr, X_te, y_te, inp_dim=input_dim, sv=seed_val):
            return run_mlp_pytorch(X_tr, y_tr, X_te, y_te, inp_dim,
                                   use_early_stopping=True, seed=sv,
                                   max_epochs=100, weight_decay=0.0)

        r = run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func_seed, f"seed_{seed_val}", input_dim)
        r["seed"] = seed_val
        sweep_results[key] = r
        with open(sweep_file, "w") as f:
            json.dump(sweep_results, f, indent=2)

        print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | Pool gain={r['gain_pooled_pp']:+.1f}pp")

    # Summary
    print(f"\n{'='*80}")
    print(f"=== M0→M5 ABLATION SUMMARY ===")
    print(f"{'Config':<30s} | {'PF gain':>8s} | {'Pool gain':>9s} | {'p':>6s}")
    print("-"*60)
    for k in ["M0_baseline", "M1_plus_seed", "M2_plus_early_stop",
              "M3_plus_weight_decay", "M4_plus_epochs200", "M5_sklearn_anchor"]:
        if k in results:
            r = results[k]
            print(f"{k:<30s} | {r['gain_perfold_pp']:>+7.1f}pp | {r['gain_pooled_pp']:>+8.1f}pp | {r['paired_p']:>6.4f}")

    print(f"\n=== SEED SWEEP SUMMARY ===")
    gains = [sweep_results[k]["gain_perfold_pp"] for k in sorted(sweep_results.keys())]
    pool_gains = [sweep_results[k]["gain_pooled_pp"] for k in sorted(sweep_results.keys())]
    if gains:
        print(f"  Per-fold gain: mean={np.mean(gains):+.1f}pp, std={np.std(gains):.1f}pp, "
              f"range=[{min(gains):+.1f}, {max(gains):+.1f}]")
        print(f"  Pooled gain:   mean={np.mean(pool_gains):+.1f}pp, std={np.std(pool_gains):.1f}pp, "
              f"range=[{min(pool_gains):+.1f}, {max(pool_gains):+.1f}]")
        seed42_idx = 42 if 42 < 20 else None
        if seed42_idx is not None and f"seed_{seed42_idx}" in sweep_results:
            print(f"  seed=42 position: PF gain={sweep_results[f'seed_{seed42_idx}']['gain_perfold_pp']:+.1f}pp")

    print(f"\n=== M0→M5 + Seed Sweep DONE ===")

if __name__ == "__main__":
    main()