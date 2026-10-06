import os  # [脱敏] 供读取 CODONBENCH_REPO_ROOT
"""
Rigorous evaluation of from-scratch synonym randomization.
- 5-fold stratified CV (per-fold delta)
- Wilcoxon signed-rank test
- Bootstrap 95% CI
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

BASE = r'os.environ.get("CODONBENCH_REPO_ROOT", "/path/to/your/codonbench") + "/results/supplementary"'

def cv_eval_paired(emb_orig, emb_rand, labs, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    fold_results = []
    for train_idx, test_idx in skf.split(emb_orig, labs):
        X_tr_o, X_te_o = emb_orig[train_idx], emb_orig[test_idx]
        X_tr_r, X_te_r = emb_rand[train_idx], emb_rand[test_idx]
        y_tr, y_te = labs[train_idx], labs[test_idx]

        lr_o = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(X_tr_o, y_tr)
        lr_r = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(X_tr_r, y_tr)
        lr_auc_o = roc_auc_score(y_te, lr_o.predict_proba(X_te_o)[:, 1])
        lr_auc_r = roc_auc_score(y_te, lr_r.predict_proba(X_te_r)[:, 1])

        mlp_o = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                              random_state=seed, early_stopping=True,
                              validation_fraction=0.1).fit(X_tr_o, y_tr)
        mlp_r = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                              random_state=seed, early_stopping=True,
                              validation_fraction=0.1).fit(X_tr_r, y_tr)
        mlp_auc_o = roc_auc_score(y_te, mlp_o.predict_proba(X_te_o)[:, 1])
        mlp_auc_r = roc_auc_score(y_te, mlp_r.predict_proba(X_te_r)[:, 1])

        fold_results.append({
            'lr_orig': lr_auc_o, 'lr_rand': lr_auc_r,
            'mlp_orig': mlp_auc_o, 'mlp_rand': mlp_auc_r,
            'd_lr': lr_auc_o - lr_auc_r,
            'd_mlp': mlp_auc_o - mlp_auc_r,
        })
    return fold_results

def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)

def wilcoxon_test(values):
    values = np.array(values)
    if np.all(values == 0):
        return 1.0
    try:
        stat, p = wilcoxon(values)
        return p
    except ValueError:
        return 1.0

def run_analysis(model, task, suffix=''):
    emb_o = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_emb.npy')
    emb_r = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_rand_emb.npy')
    labs = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_labels.npy')

    folds = cv_eval_paired(emb_o, emb_r, labs)

    d_mlp_vals = [f['d_mlp'] for f in folds]
    d_lr_vals = [f['d_lr'] for f in folds]
    mlp_orig_vals = [f['mlp_orig'] for f in folds]
    mlp_rand_vals = [f['mlp_rand'] for f in folds]

    d_mlp_pp = [d * 100 for d in d_mlp_vals]
    d_lr_pp = [d * 100 for d in d_lr_vals]

    mlp_ci_lo, mlp_ci_hi = bootstrap_ci(d_mlp_pp)
    lr_ci_lo, lr_ci_hi = bootstrap_ci(d_lr_pp)

    mlp_p = wilcoxon_test(d_mlp_vals)
    lr_p = wilcoxon_test(d_lr_vals)

    interp_mlp = 'hurts (genuine)' if np.mean(d_mlp_pp) > 0.5 else ('helps (gene-id)' if np.mean(d_mlp_pp) < -0.5 else 'no effect')

    print(f'\n=== {model} {task}{suffix} ===')
    print(f'  MLP orig:  {np.mean(mlp_orig_vals):.4f} +/- {np.std(mlp_orig_vals):.4f}')
    print(f'  MLP rand:  {np.mean(mlp_rand_vals):.4f} +/- {np.std(mlp_rand_vals):.4f}')
    print(f'  dMLP:      {np.mean(d_mlp_pp):+.1f} pp  [{mlp_ci_lo:+.1f}, {mlp_ci_hi:+.1f}]  p={mlp_p:.3f}')
    print(f'  dLR:       {np.mean(d_lr_pp):+.1f} pp  [{lr_ci_lo:+.1f}, {lr_ci_hi:+.1f}]  p={lr_p:.3f}')
    print(f'  Per-fold dMLP (pp): {[f"{v:+.1f}" for v in d_mlp_pp]}')
    print(f'  Interpretation: {interp_mlp}')

    return {
        'model': model, 'task': task,
        'mlp_orig_mean': np.mean(mlp_orig_vals),
        'mlp_rand_mean': np.mean(mlp_rand_vals),
        'd_mlp_pp': np.mean(d_mlp_pp),
        'd_mlp_ci': (mlp_ci_lo, mlp_ci_hi),
        'd_mlp_p': mlp_p,
        'd_lr_pp': np.mean(d_lr_pp),
        'd_lr_p': lr_p,
        'per_fold_d_mlp': d_mlp_pp,
        'interp': interp_mlp,
    }

if __name__ == '__main__':
    print('=== From-Scratch Synonym Randomization: Rigorous Evaluation ===')
    print('  5-fold CV | Wilcoxon signed-rank | Bootstrap 95% CI\n')

    results = []
    for suffix in ['', '_ctx128']:
        tag = f' (ctx={16 if suffix == "" else 128})'
        for model in ['v3b', 'v4']:
            for task in ['synonymous', 'missense']:
                try:
                    r = run_analysis(model, task, suffix)
                    r['context'] = 16 if suffix == '' else 128
                    results.append(r)
                except FileNotFoundError:
                    pass

    print('\n\n=== SUMMARY TABLE ===')
    print(f'{"Model":6s} {"Task":12s} {"Ctx":4s} | {"Orig MLP":9s} {"Rand MLP":9s} | {"dMLP(pp)":9s} {"95% CI":16s} {"p":6s} | {"Interp"}')
    print('-' * 95)
    for r in results:
        ci = r['d_mlp_ci']
        print(f'{r["model"]:6s} {r["task"]:12s} {r["context"]:4d} | {r["mlp_orig_mean"]:.4f}    {r["mlp_rand_mean"]:.4f}    | {r["d_mlp_pp"]:+6.1f}    [{ci[0]:+5.1f}, {ci[1]:+5.1f}]  {r["d_mlp_p"]:.3f} | {r["interp"]}')

    print('\n=== COMPARISON WITH PRETRAINED (Table S13) ===')
    print('CodonBERT  SynPath  dMLP = -5.2 pp (hurts -> genuine)')
    print('CB-HF      SynPath  dMLP = +1.2 pp (helps -> gene-id)')
    print('EC-80M     SynPath  dMLP = +1.7 pp (helps -> gene-id)')
    print('ESM-2      SynPath  dMLP =  0.0 pp (no effect)')