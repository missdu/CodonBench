import sys; sys.path.insert(0, '.')
import os; os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, json, time, torch, torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats as sp_stats

UNIFIED_DIR = Path('./results/unified_eval')
MIS_DIR = UNIFIED_DIR / 'mispath_data'
EVAL_DIR = UNIFIED_DIR / 'eval_results'
DEVICE = 'cuda:0'
RESULTS_FILE = EVAL_DIR / 'm0_m5_cross_model_results.json'

MLP_CONFIGS = [
    dict(name='M0_baseline', use_es=False, seed=None, epochs=100, wd=0.0),
    dict(name='M1_plus_seed', use_es=False, seed=42, epochs=100, wd=0.0),
    dict(name='M2_plus_early_stop', use_es=True, seed=42, epochs=100, wd=0.0),
    dict(name='M3_plus_weight_decay', use_es=True, seed=42, epochs=100, wd=1e-4),
    dict(name='M4_plus_epochs200', use_es=True, seed=42, epochs=200, wd=1e-4),
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
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    lr.fit(X_train, y_train)
    return roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

def run_mlp_sklearn(X_train, y_train, X_test, y_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        early_stopping=True, n_iter_no_change=10,
                        random_state=42, solver='adam',
                        validation_fraction=0.1)
    mlp.fit(X_train, y_train)
    return roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

def run_mlp_pytorch(X_train, y_train, X_test, y_test, config=None):
    if config is None: config = {}
    set_seed(config.get('seed'))
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    model = EarlyStoppingMLP(X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config.get('wd', 0))
    criterion = nn.BCELoss()
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    for epoch in range(config.get('epochs', 100)):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        X_te = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        prob = model(X_te).squeeze().cpu().numpy()
    return roc_auc_score(y_test, prob)

def run_logo_cv(emb, y, tx_arr, valid_txs, probe_func, **kwargs):
    lr_aucs, mlp_aucs = [], []
    for tx in valid_txs:
        mask_te = tx_arr == tx
        mask_tr = ~mask_te
        if y[mask_tr].sum() < 2 or y[mask_te].sum() < 2:
            continue
        if (1-y[mask_tr]).sum() < 2 or (1-y[mask_te]).sum() < 2:
            continue
        try:
            lr_auc = run_lr(emb[mask_tr], y[mask_tr], emb[mask_te], y[mask_te])
            mlp_auc = probe_func(emb[mask_tr], y[mask_tr], emb[mask_te], y[mask_te], **kwargs)
            lr_aucs.append(lr_auc)
            mlp_aucs.append(mlp_auc)
        except:
            continue
    if len(lr_aucs) < 10:
        return None
    lr_m = np.mean(lr_aucs)
    mlp_m = np.mean(mlp_aucs)
    t, p = sp_stats.ttest_rel(mlp_aucs, lr_aucs)
    return dict(lr_perfold_mean=round(lr_m, 4), mlp_perfold_mean=round(mlp_m, 4),
            gain_perfold_pp=round((mlp_m - lr_m) * 100, 1), n_folds=len(lr_aucs),
            paired_t=round(t, 3), paired_p=round(p, 4))

def main():
    print(f'[{time.strftime("%H:%M:%S")}] Loading CodonBERT MisPath data...')
    emb = np.load(MIS_DIR / 'codonbert_task2_missense_emb.npy')
    y = np.load(MIS_DIR / 'codonbert_task2_missense_labels.npy')
    tx_arr = np.load(MIS_DIR / 'codonbert_tx_ids.npy', allow_pickle=True)
    split = np.load(MIS_DIR / 'split_logo_cv.npz', allow_pickle=True)
    if 'valid_txs' in split:
        valid_txs = split['valid_txs']
    elif 'valid_fold_tx_ids' in split:
        valid_txs = split['valid_fold_tx_ids']
    else:
        valid_txs = np.unique(tx_arr)
    print(f'  Embedding: {emb.shape}, Labels: {y.shape}, Valid txs: {len(valid_txs)}')

    all_results = json.load(open(RESULTS_FILE)) if RESULTS_FILE.exists() else {}
    combo_key = 'codonbert_mispath'
    combo_results = all_results.get(combo_key, {})

    for cfg in MLP_CONFIGS:
        cfg_key = cfg['name']
        if cfg_key in combo_results:
            print(f'  SKIP {cfg_key} (already done)')
            continue
        print(f'  [{time.strftime("%H:%M:%S")}] --- {cfg_key} ---')
        r = run_logo_cv(emb, y, tx_arr, valid_txs, run_mlp_pytorch, config=cfg)
        if r is not None:
            r['config'] = cfg
            combo_results[cfg_key] = r
            all_results[combo_key] = combo_results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    sklearn_key = 'M5_sklearn_anchor'
    if sklearn_key not in combo_results:
        print(f'  [{time.strftime("%H:%M:%S")}] --- {sklearn_key} ---')
        r = run_logo_cv(emb, y, tx_arr, valid_txs, run_mlp_sklearn)
        if r is not None:
            r['config'] = dict(name='M5_sklearn', impl='sklearn_MLPClassifier')
            combo_results[sklearn_key] = r
            all_results[combo_key] = combo_results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  DONE | PF gain={r['gain_perfold_pp']:+.1f}pp | p={r['paired_p']:.4f}")

    m0 = combo_results.get('M0_baseline', {}).get('gain_perfold_pp', '?')
    m5 = combo_results.get('M5_sklearn_anchor', {}).get('gain_perfold_pp', '?')
    if isinstance(m0, (int, float)) and isinstance(m5, (int, float)):
        print(f'CodonBERT MisPath: M0={m0:+.1f}pp, M5={m5:+.1f}pp, diff={m5-m0:+.1f}pp')
    else:
        print(f'CodonBERT MisPath: M0={m0}, M5={m5}')

if __name__ == '__main__':
    main()