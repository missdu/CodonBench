import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

def eval_emb(emb, labs):
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labs, test_size=0.2, random_state=42, stratify=labs)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs').fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:,1])
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42, early_stopping=True, validation_fraction=0.1).fit(X_tr, y_tr)
    mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:,1])
    return lr_auc, mlp_auc

base = r'F:\2025-2026-2\CODEARTS\shorts\codonbench\codonbench\results\supplementary'

print('=== From-Scratch Synonym Randomization ===')

for model in ['v3b', 'v4']:
    for task in ['synonymous', 'missense']:
        emb_o = np.load(f'{base}/from_scratch_{model}_{task}_emb.npy')
        emb_r = np.load(f'{base}/from_scratch_{model}_{task}_rand_emb.npy')
        labs = np.load(f'{base}/from_scratch_{model}_{task}_labels.npy')
        
        lr_o, mlp_o = eval_emb(emb_o, labs)
        lr_r, mlp_r = eval_emb(emb_r, labs)
        
        d_lr = (lr_o - lr_r) * 100
        d_mlp = (mlp_o - mlp_r) * 100
        
        interp = 'hurts(genuine)' if d_mlp > 0.5 else ('helps(gene-id)' if d_mlp < -0.5 else 'no effect')
        print(f'{model:6s} {task:12s}  Orig-LR={lr_o:.4f} MLP={mlp_o:.4f}  Rand-LR={lr_r:.4f} MLP={mlp_r:.4f}  dLR={d_lr:+.1f} dMLP={d_mlp:+.1f}  {interp}')

print()
print('=== Pretrained Models (Table S13) ===')
print('CodonBERT  SynPath  dMLP = -5.2 pp (hurts -> genuine)')
print('CB-HF      SynPath  dMLP = +1.2 pp (helps -> gene-id)')
print('EC-80M     SynPath  dMLP = +1.7 pp (helps -> gene-id)')
print('ESM-2      SynPath  dMLP =  0.0 pp (no effect)')