import os  # [脱敏] 供读取 CODONBENCH_REPO_ROOT
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import warnings; warnings.filterwarnings('ignore')

BASE = r'os.environ.get("CODONBENCH_REPO_ROOT", "/path/to/your/codonbench") + "/results/supplementary"'

for ctx in [16, 128]:
    suffix = '' if ctx == 16 else '_ctx128'
    for model in ['v3b', 'v4']:
        for task in ['synonymous', 'missense']:
            try:
                emb_o = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_emb.npy')
                emb_r = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_rand_emb.npy')
                labs = np.load(f'{BASE}/from_scratch_{model}_{task}{suffix}_labels.npy')
            except FileNotFoundError:
                continue

            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            d_mlp_pp = []
            mlp_orig_folds = []
            mlp_rand_folds = []
            for train_idx, test_idx in skf.split(emb_o, labs):
                X_tr_o, X_te_o = emb_o[train_idx], emb_o[test_idx]
                X_tr_r, X_te_r = emb_r[train_idx], emb_r[test_idx]
                y_tr, y_te = labs[train_idx], labs[test_idx]
                mlp_o = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1).fit(X_tr_o, y_tr)
                mlp_r = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1).fit(X_tr_r, y_tr)
                auc_o = roc_auc_score(y_te, mlp_o.predict_proba(X_te_o)[:,1])
                auc_r = roc_auc_score(y_te, mlp_r.predict_proba(X_te_r)[:,1])
                mlp_orig_folds.append(auc_o)
                mlp_rand_folds.append(auc_r)
                d_mlp_pp.append((auc_o - auc_r) * 100)

            mean_d = np.mean(d_mlp_pp)
            try:
                _, p = wilcoxon(d_mlp_pp)
            except:
                p = 1.0
            interp = 'hurts(genuine)' if mean_d > 0.5 else ('helps(gene-id)' if mean_d < -0.5 else 'no effect')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            print(f'{model:5s} {task:12s} ctx={ctx:3d} | MLP_orig={np.mean(mlp_orig_folds):.3f} MLP_rand={np.mean(mlp_rand_folds):.3f} | dMLP={mean_d:+5.1f}pp p={p:.3f} {sig} | {interp} | folds={[f"{v:+.1f}" for v in d_mlp_pp]}')