import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("./results/supplementary")

EMB_MAP = {
    ("ESM-2-650M", "task2_missense"): "ESM-2-650M_task2_missense_emb.npy",
    ("ESM-2-650M", "task3_synonymous"): "ESM-2-650M_task3_synonymous_emb.npy",
    ("ESM-1b-650M", "task2_missense"): "ESM-1b-650M_task2_missense_emb.npy",
    ("ESM-1b-650M", "task3_synonymous"): "ESM-1b-650M_task3_synonymous_emb.npy",
    ("cdsbert", "task2_missense"): "cdsbert_task2_missense_emb.npy",
    ("cdsbert", "task3_synonymous"): "cdsbert_task3_synonymous_emb.npy",
    ("calm", "task2_missense"): "calm_task2_missense_emb.npy",
    ("calm", "task3_synonymous"): "calm_task3_synonymous_emb.npy",
}

LABEL_MAP = {
    "task2_missense": "codonbert_task2_missense_labels.npy",
    "task3_synonymous": "codonbert_task3_synonymous_labels.npy",
}

def mlp_5fold_cv(X, y, hidden_sizes=(128, 64), max_iter=100, random_state=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        mlp = MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=max_iter,
                            random_state=random_state, early_stopping=True,
                            validation_fraction=0.1, solver='adam',
                            learning_rate_init=1e-3, alpha=1e-4)
        mlp.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
        fold_aucs.append(auc)
        print(f"    Fold {fold_idx+1}: MLP AUC = {auc:.4f}")
    return np.array(fold_aucs)

def main():
    existing = []
    res_file = OUT_DIR / "mlp_5fold_cv_results.json"
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for (model, task), emb_file in EMB_MAP.items():
        if (model, task) in done_keys:
            print(f"  SKIP {model} {task}")
            continue
        emb_path = OUT_DIR / emb_file
        lab_path = OUT_DIR / LABEL_MAP[task]
        if not emb_path.exists():
            print(f"  MISSING embedding: {emb_path}")
            continue
        if not lab_path.exists():
            print(f"  MISSING labels: {lab_path}")
            continue
        print(f"\n{'='*60}\n{model} | {task}\n{'='*60}")
        emb = np.load(emb_path)
        labs = np.load(lab_path)
        n = min(len(emb), len(labs))
        emb = emb[:n]; labs = labs[:n]
        print(f"  Embeddings: {emb.shape}, Labels: {len(labs)}")

        lr_cv = cross_val_score(LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
                                emb, labs, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="roc_auc")
        print(f"  CV-LR={lr_cv.mean():.4f}±{lr_cv.std():.4f}")

        mlp_cv = mlp_5fold_cv(emb, labs)
        cv_gain = mlp_cv.mean() - lr_cv.mean()
        cv_gain_se = np.sqrt(mlp_cv.std()**2 + lr_cv.std()**2) / np.sqrt(5)

        r = {
            "model": model, "task": task, "n": int(n),
            "cv_lr_mean": round(float(lr_cv.mean()), 4),
            "cv_lr_std": round(float(lr_cv.std()), 4),
            "cv_lr_folds": [round(float(x), 4) for x in lr_cv],
            "cv_mlp_mean": round(float(mlp_cv.mean()), 4),
            "cv_mlp_std": round(float(mlp_cv.std()), 4),
            "cv_mlp_folds": [round(float(x), 4) for x in mlp_cv],
            "cv_gain": round(float(cv_gain), 4),
            "cv_gain_se": round(float(cv_gain_se), 4),
        }
        print(f"  CV-MLP={r['cv_mlp_mean']:.4f}±{r['cv_mlp_std']:.4f}")
        print(f"  Gain={r['cv_gain']:.4f}±{r['cv_gain_se']:.4f}")
        existing.append(r)
        with open(res_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:20s} | {r['task']:25s} | CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | CV-MLP={r['cv_mlp_mean']:.4f}±{r['cv_mlp_std']:.4f} | Gain={r['cv_gain']:.4f}±{r['cv_gain_se']:.4f}")

if __name__ == "__main__":
    main()