"""Step 7: onehot_pos MLP upper bound evaluation"""
import sys, os, json, re, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/onehot_pos_nonlinear")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_onehot_pos(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    codons = dna_to_codons(cds[s*3:e*3])
    if not codons: return None
    feat = np.zeros(len(codons) * 64, dtype=np.float32)
    for j, c in enumerate(codons):
        if c in CODON_TO_IDX:
            feat[j * 64 + CODON_TO_IDX[c]] = 1.0
    return feat

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden2, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_test, y_test, input_dim, epochs=100, lr=1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPProbe(input_dim).to(device)
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_v = torch.FloatTensor(X_test).to(device)
    y_v = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = Adam(model.parameters(), lr=lr)
    best_auc, best_state = 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.BCELoss()(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_v).cpu().numpy().flatten()
        auc = roc_auc_score(y_test, pred)
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep+1) % 20 == 0:
            print(f"    Epoch {ep+1}: AUC = {auc:.4f} (best = {best_auc:.4f})")
    if best_state:
        model.load_state_dict(best_state)
    return best_auc

if __name__ == "__main__":
    print("Loading data...")
    cds = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    pk, bk = ["Pathogenic", "Likely pathogenic"], ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    valid = snv[snv["label"] >= 0].copy()
    valid = valid[(valid["ReferenceAlleleVCF"].str.len()==1) & (valid["AlternateAlleleVCF"].str.len()==1)]
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))

    results = {}
    for task_name, task_df, max_n in [("task2_missense", variants[~variants["is_syn"]], 5000),
                                       ("task3_synonymous", variants[variants["is_syn"]], 2840)]:
        print(f"\n=== onehot_pos on {task_name} ===")
        df = task_df.copy()
        n_pc = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
        df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                        df[df["label"]==0].sample(n_pc, random_state=42)])

        feats, labels = [], []
        for _, row in df.iterrows():
            f = build_onehot_pos(cds[row["tx_id"]], row["cpos"])
            if f is not None: feats.append(f); labels.append(row["label"])
        X = np.array(feats)
        y = np.array(labels)
        print(f"  {len(X)} samples, feature dim = {X.shape[1]}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        lr.fit(X_train, y_train)
        lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        print(f"  LR AUC = {lr_auc:.4f}")

        print("  Training MLP...")
        mlp_auc = train_mlp(X_train, y_train, X_test, y_test, X.shape[1])
        print(f"  MLP AUC = {mlp_auc:.4f}")

        results[task_name] = {
            "model": "onehot_pos",
            "task": task_name,
            "n_samples": len(X),
            "feature_dim": X.shape[1],
            "lr_auc_test": float(lr_auc),
            "mlp_auc_test": float(mlp_auc),
            "mlp_minus_lr": float(mlp_auc - lr_auc),
        }

    with open(OUT_DIR / "onehot_pos_nonlinear.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'onehot_pos_nonlinear.json'}")