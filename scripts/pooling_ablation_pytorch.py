"""
Pooling ablation with PyTorch MLP: mean-pool vs CLS.
CodonBERT SynPath: single 80/20 split + LOGO-CV 240 folds.

Purpose: Determine whether the pooling gate effect (CLS→mean: +13.9→−5.2 pp under sklearn)
survives under PyTorch MLP. If CLS gain ≈ 0 and mean gain ≈ 0 under PyTorch,
the pooling gate narrative in R3 needs major revision.

Uses existing embeddings:
  CLS:  results/supplementary/codonbert_task3_synonymous_emb.npy
  Mean: results/supplementary/codonbert_task3_synonymous_mean_pool_emb.npy
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, json, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy import stats as sp_stats

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
EMB_DIR_SUPP = Path("./results/supplementary")
EVAL_DIR = UNIFIED_DIR / "eval_results"
DEVICE = "cuda:1"

class PyTorchMLP(nn.Module):
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

def run_lr(X_train, y_train, X_test, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob), y_prob

def run_mlp_pytorch(X_train, y_train, X_test, y_test, input_dim,
                    seed=None, max_epochs=100, weight_decay=0.0):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    model = PyTorchMLP(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    model.train()
    for ep in range(max_epochs):
        opt.zero_grad()
        pred = model(X_tr_t).squeeze()
        loss_fn(pred, y_tr_t).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        y_prob = model(X_te_t).squeeze().cpu().numpy()
    return roc_auc_score(y_test, y_prob), y_prob

def run_logo_cv(emb, y, tx_arr, valid_txs, input_dim, seed=None, max_epochs=100, wd=0.0):
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
            mlp_auc, mlp_prob = run_mlp_pytorch(emb[tr], y_tr, emb[te], y_te, input_dim,
                                                  seed=seed, max_epochs=max_epochs, weight_decay=wd)
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

def run_single_split(emb, y, input_dim, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        emb, y, test_size=0.2, random_state=42, stratify=y)

    lr_auc, _ = run_lr(X_train, y_train, X_test, y_test)
    mlp_auc, _ = run_mlp_pytorch(X_train, y_train, X_test, y_test, input_dim,
                                   seed=seed, max_epochs=100)
    return {
        "lr_auc": round(float(lr_auc), 4),
        "mlp_auc": round(float(mlp_auc), 4),
        "gain_pp": round((mlp_auc - lr_auc) * 100, 1),
    }

def main():
    print(f"[{time.strftime('%H:%M:%S')}] === Pooling Ablation PyTorch MLP START ===")
    print(f"[{time.strftime('%H:%M:%S')}] Device: {DEVICE}")

    with open(SYN_DIR / "sequences.json") as f:
        seq_data = json.load(f)
    labels = np.array(seq_data["labels"])
    tx_ids = np.array(seq_data["tx_ids"])

    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])

    results = {}

    for pool_name, emb_file in [
        ("CLS", "codonbert_task3_synonymous_emb.npy"),
        ("mean", "codonbert_task3_synonymous_mean_pool_emb.npy"),
    ]:
        emb_path = EMB_DIR_SUPP / emb_file
        if not emb_path.exists():
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: {emb_path} not found!")
            continue

        emb = np.load(emb_path)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {pool_name} pooling ===")
        print(f"  Embedding shape: {emb.shape}")

        n = min(len(emb), len(labels))
        emb = emb[:n]; y = labels[:n]; tx_arr = tx_ids[:n]
        input_dim = emb.shape[1]

        # Single 80/20 split
        print(f"[{time.strftime('%H:%M:%S')}] Running single 80/20 split...")
        single_result = run_single_split(emb, y, input_dim)
        print(f"  LR={single_result['lr_auc']:.4f} MLP={single_result['mlp_auc']:.4f} "
              f"Gain={single_result['gain_pp']:+.1f}pp")

        # LOGO-CV
        print(f"[{time.strftime('%H:%M:%S')}] Running LOGO-CV (240 folds)...")
        logo_result = run_logo_cv(emb, y, tx_arr, valid_txs, input_dim, seed=42, max_epochs=100)
        print(f"  LOGO PF gain={logo_result['gain_perfold_pp']:+.1f}pp | "
              f"Pool gain={logo_result['gain_pooled_pp']:+.1f}pp | "
              f"p={logo_result['paired_p']:.4f}")

        results[pool_name] = {
            "single_split": single_result,
            "logo_cv": logo_result,
            "embedding_file": emb_file,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"=== POOLING ABLATION PyTorch MLP SUMMARY ===")
    print(f"{'Pooling':<8s} | {'Single LR':>9s} | {'Single MLP':>10s} | {'Single Gain':>11s} | "
          f"{'LOGO PF Gain':>12s} | {'LOGO Pool Gain':>14s} | {'p':>6s}")
    print("-"*80)
    for pool_name in ["CLS", "mean"]:
        if pool_name in results:
            r = results[pool_name]
            s = r["single_split"]
            l = r["logo_cv"]
            print(f"{pool_name:<8s} | {s['lr_auc']:>9.4f} | {s['mlp_auc']:>10.4f} | "
                  f"{s['gain_pp']:>+10.1f}pp | {l['gain_perfold_pp']:>+11.1f}pp | "
                  f"{l['gain_pooled_pp']:>+13.1f}pp | {l['paired_p']:>6.4f}")

    # Gate effect
    if "CLS" in results and "mean" in results:
        cls_single_gain = results["CLS"]["single_split"]["gain_pp"]
        mean_single_gain = results["mean"]["single_split"]["gain_pp"]
        cls_logo_gain = results["CLS"]["logo_cv"]["gain_perfold_pp"]
        mean_logo_gain = results["mean"]["logo_cv"]["gain_perfold_pp"]

        print(f"\n=== GATE EFFECT ===")
        print(f"  Single split: CLS→mean = {cls_single_gain:+.1f} → {mean_single_gain:+.1f} pp "
              f"(shift = {mean_single_gain - cls_single_gain:+.1f} pp)")
        print(f"  LOGO-CV:      CLS→mean = {cls_logo_gain:+.1f} → {mean_logo_gain:+.1f} pp "
              f"(shift = {mean_logo_gain - cls_logo_gain:+.1f} pp)")
        if abs(cls_logo_gain) < 1.0 and abs(mean_logo_gain) < 1.0:
            print(f"  ⚠️ Both gains near zero under PyTorch MLP → pooling gate effect DISAPPEARS")
        else:
            print(f"  ✅ Pooling gate effect persists under PyTorch MLP")

    # Save
    out_file = EVAL_DIR / "pooling_ablation_pytorch_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print(f"[{time.strftime('%H:%M:%S')}] === Pooling Ablation PyTorch MLP DONE ===")

if __name__ == "__main__":
    main()