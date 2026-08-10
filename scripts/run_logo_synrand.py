"""
LOGO-CV Synonym Randomization Evaluation
Uses cached embeddings from results/supplementary/
Reconstructs tx_ids from ClinVar data to ensure correct LOGO fold assignment.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, time, random, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from collections import defaultdict

BASE = Path(".")
SUPP = BASE / "results" / "supplementary"
UNIFIED = BASE / "results" / "unified_eval"
EVAL_DIR = UNIFIED / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE / "data"

LOG_FILE = EVAL_DIR / "logo_synrand.log"
OUT_FILE = EVAL_DIR / "logo_synrand_results.json"

CODON_CONTEXT = 16

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TABLE.items():
    if aa != '*':
        AA_TO_CODONS[aa].append(codon)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)


def load_clinvar_data():
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
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds.keys()))
    var = v[h].copy()
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    return cds, var


def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        return pd.concat([df[df["label"] == 1].sample(n, random_state=42),
                          df[df["label"] == 0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df


def build_logo_folds(tx_ids, labels, min_test_size=2):
    unique_txs = sorted(set(tx_ids))
    folds = []
    for tx in unique_txs:
        test_mask = tx_ids == tx
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        y_test = labels[test_idx]
        y_train = labels[train_idx]
        if len(set(y_train)) < 2 or len(set(y_test)) < 2 or len(test_idx) < min_test_size:
            continue
        folds.append((tx, train_idx, test_idx))
    return folds


def run_lr_logo(emb, labels, folds, solver="lbfgs", use_scaler=True):
    aucs = []
    for tx_id, train_idx, test_idx in folds:
        X_train, y_train = emb[train_idx], labels[train_idx]
        X_test, y_test = emb[test_idx], labels[test_idx]
        if use_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver=solver)
        lr.fit(X_train, y_train)
        auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        aucs.append(auc)
    if not aucs:
        return None, None, 0
    return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4), len(aucs)


def run_mlp_logo(emb, labels, folds, use_scaler=True):
    import torch
    import torch.nn as nn

    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    aucs = []

    for fold_i, (tx_id, train_idx, test_idx) in enumerate(folds):
        X_train, y_train = emb[train_idx], labels[train_idx]
        X_test, y_test = emb[test_idx], labels[test_idx]

        if use_scaler:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
        else:
            X_train_s = X_train
            X_test_s = X_test

        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)

        n_features = X_train_s.shape[1]
        mlp = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        n_epochs = 300
        batch_size = 256
        n_train = len(y_train)
        best_loss = float("inf")
        patience = 20
        wait = 0

        for epoch in range(n_epochs):
            mlp.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]
                pred = mlp(xb).squeeze()
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n_train

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        mlp.eval()
        with torch.no_grad():
            pred = mlp(X_test_t).squeeze().cpu().numpy()
        auc = roc_auc_score(y_test, pred)
        aucs.append(auc)

        if (fold_i + 1) % 50 == 0:
            log(f"    MLP fold {fold_i+1}/{len(folds)}, running mean AUC={np.mean(aucs):.4f}")

        torch.cuda.empty_cache()

    if not aucs:
        return None, None, 0
    return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4), len(aucs)


def main():
    log("=== LOGO-CV Synonym Randomization Evaluation START ===")

    cds_cache, variants = load_clinvar_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    MODELS_SYNPATH = [
        ("codonbert", "codonbert_synonymous", "codonbert_synonymous_rand_all"),
        ("codonbert_hf", "codonbert_hf_synonymous", "codonbert_hf_synonymous_rand"),
        ("encodon-80m", "encodon-80m_synonymous", "encodon-80m_synonymous_rand"),
        ("esm2", "esm2_synonymous_protein", "esm2_synonymous_protein_rand"),
    ]

    MODELS_MISPATH = [
        ("codonbert", "codonbert_missense", "codonbert_missense_rand_all"),
        ("codonbert_hf", "codonbert_hf_missense", "codonbert_hf_missense_rand"),
        ("encodon-80m", "encodon-80m_missense", "encodon-80m_missense_rand"),
        ("esm2", "esm2_missense_protein", "esm2_missense_protein_rand"),
    ]

    results = []

    for task, model_list, task_df, max_n in [
        ("synonymous", MODELS_SYNPATH, t3, max3),
        ("missense", MODELS_MISPATH, t2, 5000),
    ]:
        log(f"\n{'='*60}")
        log(f"Task: {task}")
        log(f"{'='*60}")

        df = balance(task_df.copy(), max_n)
        tx_ids = df["tx_id"].values
        labels = df["label"].values
        tx_ids_arr = np.array(tx_ids)

        folds = build_logo_folds(tx_ids_arr, labels)
        log(f"  {task}: {len(labels)} samples, {len(folds)} valid LOGO folds")

        for model_name, orig_key, rand_key in model_list:
            log(f"\n--- {model_name} | {task} ---")

            orig_path = SUPP / f"{orig_key}_emb.npy"
            rand_path = SUPP / f"{rand_key}_emb.npy"

            if not orig_path.exists():
                log(f"  SKIP: {orig_path} not found")
                continue
            if not rand_path.exists():
                log(f"  SKIP: {rand_path} not found (no synrand embedding)")
                continue

            emb_orig = np.load(orig_path)
            emb_rand = np.load(rand_path)

            if len(emb_orig) != len(labels):
                log(f"  SKIP: emb_orig {len(emb_orig)} != labels {len(labels)}")
                continue
            if len(emb_rand) != len(labels):
                log(f"  SKIP: emb_rand {len(emb_rand)} != labels {len(labels)}")
                continue

            log(f"  orig emb: {emb_orig.shape}, rand emb: {emb_rand.shape}")

            # LR on original
            log(f"  Running LR LOGO-CV on ORIGINAL...")
            orig_lr_mean, orig_lr_std, orig_lr_n = run_lr_logo(emb_orig, labels, folds)
            log(f"  Original LR LOGO: {orig_lr_mean:.4f} +/- {orig_lr_std:.4f} (n={orig_lr_n})")

            # LR on randomized
            log(f"  Running LR LOGO-CV on RANDOMIZED...")
            rand_lr_mean, rand_lr_std, rand_lr_n = run_lr_logo(emb_rand, labels, folds)
            log(f"  Randomized LR LOGO: {rand_lr_mean:.4f} +/- {rand_lr_std:.4f} (n={rand_lr_n})")

            # MLP on original
            log(f"  Running MLP LOGO-CV on ORIGINAL...")
            orig_mlp_mean, orig_mlp_std, orig_mlp_n = run_mlp_logo(emb_orig, labels, folds)
            log(f"  Original MLP LOGO: {orig_mlp_mean:.4f} +/- {orig_mlp_std:.4f} (n={orig_mlp_n})")

            # MLP on randomized
            log(f"  Running MLP LOGO-CV on RANDOMIZED...")
            rand_mlp_mean, rand_mlp_std, rand_mlp_n = run_mlp_logo(emb_rand, labels, folds)
            log(f"  Randomized MLP LOGO: {rand_mlp_mean:.4f} +/- {rand_mlp_std:.4f} (n={rand_mlp_n})")

            delta_lr = round(orig_lr_mean - rand_lr_mean, 4) if orig_lr_mean and rand_lr_mean else None
            delta_mlp = round(orig_mlp_mean - rand_mlp_mean, 4) if orig_mlp_mean and rand_mlp_mean else None

            entry = {
                "model": model_name,
                "task": task,
                "mode": "all_codons",
                "cv_type": "LOGO",
                "original_lr_mean": orig_lr_mean,
                "original_lr_std": orig_lr_std,
                "original_mlp_mean": orig_mlp_mean,
                "original_mlp_std": orig_mlp_std,
                "randomized_lr_mean": rand_lr_mean,
                "randomized_lr_std": rand_lr_std,
                "randomized_mlp_mean": rand_mlp_mean,
                "randomized_mlp_std": rand_mlp_std,
                "delta_lr": delta_lr,
                "delta_mlp": delta_mlp,
                "n_folds_lr": orig_lr_n,
                "n_folds_mlp": orig_mlp_n,
            }
            results.append(entry)

            log(f"\n  >>> DELTA: LR={delta_lr:+.4f}, MLP={delta_mlp:+.4f}")

            with open(OUT_FILE, "w") as f:
                json.dump(results, f, indent=2, default=str)

    log(f"\n{'='*60}")
    log("ALL DONE")
    log(f"{'='*60}")
    for r in results:
        dl = f"{r['delta_lr']:+.4f}" if r['delta_lr'] is not None else "N/A"
        dm = f"{r['delta_mlp']:+.4f}" if r['delta_mlp'] is not None else "N/A"
        log(f"  {r['model']:15s} | {r['task']:12s} | dLR={dl} | dMLP={dm}")


if __name__ == "__main__":
    main()
