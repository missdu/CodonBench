"""
Compute LR (single 80/20 split) for from-scratch models.
Uses the same random_state=42 as the MLP single split,
so LR and MLP are on identical train/test splits.
"""
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pathlib import Path

SUPP = Path("results/supplementary")

def lr_single_split(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver='liblinear')
    lr.fit(X_train, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    return auc

def lr_5fold_cv(X, y, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='liblinear')
        lr.fit(X_train, y_train)
        auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        fold_aucs.append(auc)
    return np.mean(fold_aucs), np.std(fold_aucs), fold_aucs

results = {}

for cond in ["v3b", "v4"]:
    for tok in ["codon", "char"]:
        key = f"{tok}-{cond}"
        emb_path = SUPP / f"from_scratch_{cond}_synonymous_emb.npy"
        lab_path = SUPP / f"from_scratch_{cond}_synonymous_labels.npy"
        
        if not emb_path.exists() or not lab_path.exists():
            print(f"  SKIP {key}: files not found")
            continue
        
        X = np.load(emb_path)
        y = np.load(lab_path)
        print(f"{key}: X={X.shape}, y={y.shape}, pos={y.sum()}, neg={len(y)-y.sum()}")
        
        lr_8020 = lr_single_split(X, y)
        lr_5f_mean, lr_5f_std, lr_5f_folds = lr_5fold_cv(X, y)
        
        results[key] = {
            "lr_single_8020": round(float(lr_8020), 4),
            "lr_5fold_mean": round(float(lr_5f_mean), 4),
            "lr_5fold_std": round(float(lr_5f_std), 4),
            "lr_5fold_folds": [round(float(a), 4) for a in lr_5f_folds],
            "n": len(y),
        }
        print(f"  LR (80/20): {lr_8020:.4f}")
        print(f"  LR (5-fold): {lr_5f_mean:.4f} +/- {lr_5f_std:.4f}")

out_path = SUPP / "from_scratch_lr_single_split_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

# Also load existing MLP results for comparison
with open(SUPP / "from_scratch_ablation_all_conditions.json") as f:
    all_cond = json.load(f)

print("\n\n=== COMPARISON: LR (80/20) vs LR (5-fold) vs MLP (80/20) ===")
for key, r in results.items():
    cond = key.split("-")[1]
    tok = key.split("-")[0]
    cond_data = all_cond.get("data_scale_layer", {}).get(cond, all_cond.get("model_scale_layer", {}).get(cond, {}))
    tok_data = cond_data.get(tok, {})
    mlp_8020 = tok_data.get("synpath_mlp", None)
    lr_5fold = tok_data.get("synpath_lr_mean", None)
    
    if mlp_8020 is not None:
        gain_8020 = (mlp_8020 - r["lr_single_8020"]) * 100
        gain_5fold = (mlp_8020 - lr_5fold) * 100 if lr_5fold else None
        print(f"  {key:15s}: LR(80/20)={r['lr_single_8020']:.3f}  LR(5f)={r['lr_5fold_mean']:.3f}  MLP(80/20)={mlp_8020:.3f}  "
              f"gain(80/20)={gain_8020:+.1f}pp  gain(5f→80/20)={gain_5fold:+.1f}pp")
    else:
        print(f"  {key:15s}: LR(80/20)={r['lr_single_8020']:.3f}  LR(5f)={r['lr_5fold_mean']:.3f}  MLP(80/20)=N/A")