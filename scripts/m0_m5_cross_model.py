"""
E6: Cross-model cross-task M0->M5 ablation.
Proves the sklearn artifact is systematic, not CodonBERT-specific.

Models: CodonBERT, EnCodon-80M, ESM-2-650M
Tasks:  SynPath (task3), MisPath (task2)
Output: m0_m5_cross_model_results.json

For each model x task, runs M0 (PyTorch baseline) and M5 (sklearn anchor).
Optionally runs full M0->M5 staircase.
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
MIS_DIR = UNIFIED_DIR / "mispath_data"
EMB_DIR = Path("./results/supplementary")
EVAL_DIR = UNIFIED_DIR / "eval_results"
DEVICE = "cuda:1"

MODELS_TASKS = [
    {"model": "codonbert", "task": "synpath", "emb_file": "codonbert_task3_synonymous_emb.npy",
     "data_dir": SYN_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
    {"model": "codonbert", "task": "mispath", "emb_file": "codonbert_task2_missense_emb.npy",
     "data_dir": MIS_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
    {"model": "encodon-80m", "task": "synpath", "emb_file": "encodon-80m_task3_synonymous_emb.npy",
     "data_dir": SYN_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
    {"model": "encodon-80m", "task": "mispath", "emb_file": "encodon-80m_task2_missense_emb.npy",
     "data_dir": MIS_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
    {"model": "esm2-650m", "task": "synpath", "emb_file": "ESM-2-650M_task3_synonymous_emb.npy",
     "data_dir": SYN_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
    {"model": "esm2-650m", "task": "mispath", "emb_file": "ESM-2-650M_task2_missense_emb.npy",
     "data_dir": MIS_DIR, "label_key": "labels", "tx_key": "tx_ids", "split_file": "split_logo_cv.npz"},
]

MLP_CONFIGS = [
    {"name": "M0_baseline", "use_es": False, "seed": None, "epochs": 100, "wd": 0.0},
    {"name": "M1_plus_seed", "use_es": False, "seed": 42, "epochs": 100, "wd": 0.0},
    {"name": "M2_plus_early_stop", "use_es": True, "seed": 42, "epochs": 100, "wd": 0.0},
    {"name": "M3_plus_weight_decay", "use_es": True, "seed": 42, "epochs": 100, "wd": 1e-4},
    {"name": "M4_plus_epochs200", "use_es": True, "seed": 42, "epochs": 200, "wd": 1e-4},
]

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
        hidden_layer_sizes=(128, 64), max_iter=200, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        solver='adam', learning_rate_init=1e-3, alpha=1e-4,
    )
    mlp.fit(X_train, y_train)
    y_prob = mlp.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, input_dim=None):
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

    if not lr_folds:
        return None
    lr_pf = float(np.mean(lr_folds))
    mlp_pf = float(np.mean(mlp_folds))
    gain_pf = (mlp_pf - lr_pf) * 100
    lr_pool = roc_auc_score(y_all, lr_preds_all)
    mlp_pool = roc_auc_score(y_all, mlp_preds_all)
    gain_pool = (mlp_pool - lr_pool) * 100
    t_stat, p_val = sp_stats.ttest_rel(mlp_folds, lr_folds)
    return {
        "lr_perfold_mean": round(lr_pf, 4),
        "mlp_perfold_mean": round(mlp_pf, 4),
        "gain_perfold_pp": round(gain_pf, 1),
        "lr_pooled": round(lr_pool, 4),
        "mlp_pooled": round(mlp_pool, 4),
        "gain_pooled_pp": round(gain_pool, 1),
        "n_folds": len(lr_folds),
        "paired_p": round(float(p_val), 4),
    }

def main():
    print(f"[{time.strftime('%H:%M:%S')}] === E6: Cross-Model Cross-Task M0->M5 Ablation START ===")

    results_file = EVAL_DIR / "m0_m5_cross_model_results.json"
    all_results = {}
    if results_file.exists():
        all_results = json.load(open(results_file))

    for mt in MODELS_TASKS:
        model_name = mt["model"]
        task_name = mt["task"]
        combo_key = f"{model_name}_{task_name}"

        print(f"\n{'='*80}")
        print(f"[{time.strftime('%H:%M:%S')}] === {combo_key} ===")

        emb_path = EMB_DIR / mt["emb_file"]
        if not emb_path.exists():
            print(f"  SKIP: {emb_path} not found")
            continue

        emb = np.load(emb_path)
        print(f"  Embedding: {emb.shape}")

        data_dir = mt["data_dir"]
        seq_file = data_dir / "sequences.json"
        split_file = data_dir / mt["split_file"]

        if not seq_file.exists() or not split_file.exists():
            print(f"  SKIP: data files not found in {data_dir}")
            continue

        with open(seq_file) as f:
            seq_data = json.load(f)
        labels = np.array(seq_data[mt["label_key"]])
        tx_ids = np.array(seq_data[mt["tx_key"]])

        logo = np.load(split_file, allow_pickle=True)
        valid_txs = list(logo["valid_fold_tx_ids"])

        n = min(len(emb), len(labels))
        emb = emb[:n]; y = labels[:n]; tx_arr = tx_ids[:n]
        input_dim = emb.shape[1]

        if combo_key not in all_results:
            all_results[combo_key] = {}

        combo_results = all_results[combo_key]

        for cfg in MLP_CONFIGS:
            cfg_key = cfg["name"]
            if cfg_key in combo_results:
                print(f"  SKIP {cfg_key}")
                continue

            print(f"\n  [{time.strftime('%H:%M:%S')}] --- {cfg_key} ---")

            def mlp_func(X_tr, y_tr, X_te, y_te, inp_dim=input_dim,
                         use_es=cfg["use_es"], seed=cfg["seed"],
                         epochs=cfg["epochs"], wd=cfg["wd"]):
                return run_mlp_pytorch(X_tr, y_tr, X_te, y_te, inp_dim,
                                       use_early_stopping=use_es, seed=seed,
                                       max_epochs=epochs, weight_decay=wd)

            r = run_logo_cv(emb, y, tx_arr, valid_txs, mlp_func, input_dim)
            if r is not None:
                r["config"] = cfg
                combo_results[cfg_key] = r
                all_results[combo_key] = combo_results
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | p={r['paired_p']:.4f}")

        sklearn_key = "M5_sklearn_anchor"
        if sklearn_key not in combo_results:
            print(f"\n  [{time.strftime('%H:%M:%S')}] --- {sklearn_key} ---")
            def sklearn_func(X_tr, y_tr, X_te, y_te):
                return run_mlp_sklearn(X_tr, y_tr, X_te, y_te)
            r = run_logo_cv(emb, y, tx_arr, valid_txs, sklearn_func)
            if r is not None:
                r["config"] = {"name": "M5_sklearn", "impl": "sklearn_MLPClassifier"}
                combo_results[sklearn_key] = r
                all_results[combo_key] = combo_results
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    print(f"\n{'='*80}")
    print(f"=== E6 CROSS-MODEL SUMMARY ===")
    print(f"{'Model_Task':<30s} | {'M0 gain':>8s} | {'M5 gain':>8s} | {'M5-M0':>7s}")
    print("-"*60)
    for combo_key in sorted(all_results.keys()):
        cr = all_results[combo_key]
        m0 = cr.get("M0_baseline", {}).get("gain_perfold_pp", "?")
        m5 = cr.get("M5_sklearn_anchor", {}).get("gain_perfold_pp", "?")
        diff = f"{m5-m0:+.1f}" if isinstance(m0, (int,float)) and isinstance(m5, (int,float)) else "?"
        m0s = f"{m0:+.1f}" if isinstance(m0, (int,float)) else str(m0)
        m5s = f"{m5:+.1f}" if isinstance(m5, (int,float)) else str(m5)
        print(f"{combo_key:<30s} | {m0s:>8s} | {m5s:>8s} | {diff:>7s}")

    print(f"\n=== E6 DONE ===")

if __name__ == "__main__":
    main()