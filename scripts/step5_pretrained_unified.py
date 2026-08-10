"""
Step 5: Unified evaluation for pretrained models.
Uses saved embeddings (CLS/mean pooled, original extraction) + unified data splits.
Covers SynPath + MisPath, single split / 5-fold CV / LOGO-CV, LR + MLP.
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, time, torch
import torch.nn as nn
import pandas as pd, re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
MIS_DIR = UNIFIED_DIR / "mispath_data"
EMB_DIR_SUPP = Path("./results/supplementary")
EMB_DIR_UNI = UNIFIED_DIR / "embeddings" / "pretrained"
EVAL_DIR = UNIFIED_DIR / "eval_results"
DATA_DIR = Path("./data")
DEVICE = "cuda:1"

PRETRAINED_MODELS = [
    ("CodonBERT", "codonbert", "codon", "CLS"),
    ("CodonBERT-HF", "codonbert_hf", "codon", "CLS"),
    ("EnCodon-80M", "encodon-80m", "codon", "CLS"),
    ("mRNABERT", "mrnabert", "codon", "mean"),
    ("ESM-2-650M", "ESM-2-650M", "protein", "mean"),
    ("ESM-1b-650M", "ESM-1b-650M", "protein", "mean"),
]

TASKS = [
    ("task3_synonymous", "SynPath", 2840),
    ("task2_missense", "MisPath", 5000),
]

LOG_FILE = EVAL_DIR / "pretrained_unified.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

def run_lr(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    return roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

def run_mlp(X_train, y_train, X_test, y_test, input_dim):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    mlp = nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    mlp.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred = mlp(X_train_t).squeeze()
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        y_pred = mlp(X_test_t).squeeze().cpu().numpy()
    return roc_auc_score(y_test, y_pred)

def load_synpath_data(emb_prefix):
    lab_path = EMB_DIR_SUPP / f"{emb_prefix}_task3_synonymous_labels.npy"
    if lab_path.exists():
        labels = np.load(lab_path)
    else:
        with open(SYN_DIR / "sequences.json") as f:
            seq_data = json.load(f)
        labels = np.array(seq_data["labels"])
    with open(SYN_DIR / "sequences.json") as f:
        seq_data = json.load(f)
    tx_ids = np.array(seq_data["tx_ids"])
    sp = np.load(SYN_DIR / "split_single_8020.npz", allow_pickle=True)
    single_train, single_test = sp["train_idx"], sp["test_idx"]
    cv = np.load(SYN_DIR / "split_5fold_cv.npz", allow_pickle=True)
    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    return labels, tx_ids, single_train, single_test, cv, valid_txs

def load_mispath_data(emb_prefix):
    MIS_DIR.mkdir(parents=True, exist_ok=True)
    lab_path = EMB_DIR_SUPP / f"{emb_prefix}_task2_missense_labels.npy"
    if lab_path.exists():
        labels = np.load(lab_path)
        tx_ids_path = MIS_DIR / f"{emb_prefix}_tx_ids.npy"
        if tx_ids_path.exists():
            tx_ids = np.load(tx_ids_path, allow_pickle=True)
        else:
            log(f"  WARNING: no tx_ids for {emb_prefix} MisPath, building from ClinVar...")
            labels, tx_ids = _build_mispath_data()
            np.save(tx_ids_path, tx_ids)
    else:
        labels, tx_ids = _build_mispath_data()

    sp_path = MIS_DIR / f"{emb_prefix}_split_single_8020.npz"
    if sp_path.exists():
        sp = np.load(sp_path, allow_pickle=True)
        single_train, single_test = sp["train_idx"], sp["test_idx"]
    else:
        from sklearn.model_selection import train_test_split
        single_train, single_test = train_test_split(
            np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels)
        np.savez(sp_path, train_idx=single_train, test_idx=single_test)

    cv_path = MIS_DIR / f"{emb_prefix}_split_5fold_cv.npz"
    if cv_path.exists():
        cv = np.load(cv_path, allow_pickle=True)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_dict = {}
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            cv_dict[f"fold_{fold}_train"] = tr
            cv_dict[f"fold_{fold}_test"] = te
        np.savez(cv_path, **cv_dict)
        cv = np.load(cv_path, allow_pickle=True)

    logo_path = MIS_DIR / f"{emb_prefix}_split_logo_cv.npz"
    if logo_path.exists():
        logo = np.load(logo_path, allow_pickle=True)
        valid_txs = list(logo["valid_fold_tx_ids"])
    else:
        unique_txs = sorted(set(tx_ids))
        valid_txs = []
        for tx in unique_txs:
            mask = tx_ids == tx
            n_test = mask.sum()
            if n_test >= 2 and len(set(labels[mask])) == 2:
                valid_txs.append(tx)
        np.savez(logo_path, valid_fold_tx_ids=np.array(valid_txs))
        log(f"  MisPath LOGO ({emb_prefix}): {len(valid_txs)} valid folds")

    return labels, tx_ids, single_train, single_test, cv, valid_txs

def _build_mispath_data():
    log("  Building MisPath data from ClinVar...")
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
    def parse_hgvs(name):
        m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
        return (m.group(1), int(m.group(3))) if m else (None, None)
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds.keys()))
    var = v[h].copy()
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    missense = var[~var["is_syn"]].copy()
    n_pos = int(missense["label"].sum())
    n_neg = int(len(missense) - missense["label"].sum())
    n_sample = min(n_pos, n_neg, 2500)
    missense_bal = pd.concat([
        missense[missense["label"] == 1].sample(n_sample, random_state=42),
        missense[missense["label"] == 0].sample(n_sample, random_state=42),
    ]).sample(frac=1, random_state=42)
    labels = missense_bal["label"].values.astype(int)
    tx_ids = missense_bal["tx_id"].values
    return labels, tx_ids

def main():
    log("=== Pretrained Unified Evaluation START ===")

    results_file = EVAL_DIR / "pretrained_unified_results.json"
    existing = {}
    if results_file.exists():
        existing = json.load(open(results_file))
        log(f"Loaded {len(existing)} existing results")

    for model_name, emb_prefix, tok_type, pooling in PRETRAINED_MODELS:
        for task_key, task_label, n_expected in TASKS:
            result_key = f"{model_name}_{task_label}"
            if result_key in existing:
                log(f"SKIP {result_key} (already done)")
                continue

            emb_file = EMB_DIR_SUPP / f"{emb_prefix}_{task_key}_emb.npy"
            if not emb_file.exists():
                log(f"MISSING {result_key}: {emb_file}")
                continue

            emb = np.load(emb_file)
            log(f"\n{'='*60}\n{result_key} | pooling={pooling} emb={emb.shape}\n{'='*60}")

            if task_label == "SynPath":
                labels, tx_ids, single_train, single_test, cv, valid_txs = load_synpath_data(emb_prefix)
            else:
                labels, tx_ids, single_train, single_test, cv, valid_txs = load_mispath_data(emb_prefix)

            n = min(len(emb), len(labels))
            emb = emb[:n]
            y = labels[:n]
            input_dim = emb.shape[1]

            # === Single 80/20 split ===
            tr = single_train[single_train < n]
            te = single_test[single_test < n]
            lr_s = run_lr(emb[tr], y[tr], emb[te], y[te])
            mlp_s = run_mlp(emb[tr], y[tr], emb[te], y[te], input_dim)
            log(f"  Single: LR={lr_s:.3f}, MLP={mlp_s:.3f}, gain={((mlp_s-lr_s)*100):+.1f}pp")

            # === 5-fold CV ===
            lr_folds, mlp_folds = [], []
            for fold in range(5):
                tr = cv[f"fold_{fold}_train"]
                te = cv[f"fold_{fold}_test"]
                tr = tr[tr < n]
                te = te[te < n]
                lr_folds.append(run_lr(emb[tr], y[tr], emb[te], y[te]))
                mlp_folds.append(run_mlp(emb[tr], y[tr], emb[te], y[te], input_dim))
                log(f"  Fold {fold}: LR={lr_folds[-1]:.3f}, MLP={mlp_folds[-1]:.3f}")

            # === LOGO-CV ===
            tx_arr = tx_ids[:n] if len(tx_ids) >= n else tx_ids
            lr_logo, mlp_logo = [], []
            log(f"  Running LOGO-CV ({len(valid_txs)} folds)...")
            for fi, tx in enumerate(valid_txs):
                test_mask = tx_arr == tx
                train_mask = ~test_mask
                tr = np.where(train_mask)[0]
                te = np.where(test_mask)[0]
                y_tr, y_te = y[tr], y[te]
                if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
                    continue
                try:
                    lr_auc = run_lr(emb[tr], y_tr, emb[te], y_te)
                    mlp_auc = run_mlp(emb[tr], y_tr, emb[te], y_te, input_dim)
                    lr_logo.append(lr_auc)
                    mlp_logo.append(mlp_auc)
                except:
                    continue
                if (fi + 1) % 50 == 0:
                    log(f"    LOGO fold {fi+1}/{len(valid_txs)}, LR={np.mean(lr_logo):.3f}, MLP={np.mean(mlp_logo):.3f}")

            existing[result_key] = {
                "model": model_name,
                "task": task_label,
                "pooling": pooling,
                "tokenizer_type": tok_type,
                "n": int(n),
                "lr_single": round(float(lr_s), 4),
                "mlp_single": round(float(mlp_s), 4),
                "gain_single_pp": round((mlp_s - lr_s) * 100, 1),
                "lr_5fold_mean": round(float(np.mean(lr_folds)), 4),
                "lr_5fold_folds": [round(float(x), 4) for x in lr_folds],
                "mlp_5fold_mean": round(float(np.mean(mlp_folds)), 4),
                "mlp_5fold_folds": [round(float(x), 4) for x in mlp_folds],
                "gain_5fold_pp": round((np.mean(mlp_folds) - np.mean(lr_folds)) * 100, 1),
                "lr_logo_mean": round(float(np.mean(lr_logo)), 4) if lr_logo else None,
                "mlp_logo_mean": round(float(np.mean(mlp_logo)), 4) if mlp_logo else None,
                "gain_logo_pp": round((np.mean(mlp_logo) - np.mean(lr_logo)) * 100, 1) if mlp_logo and lr_logo else None,
                "lr_logo_n_folds": len(lr_logo),
                "mlp_logo_n_folds": len(mlp_logo),
            }

            with open(results_file, "w") as f:
                json.dump(existing, f, indent=2)
            log(f"  Saved result for {result_key}")

            del emb
            torch.cuda.empty_cache()

    log("\n=== SUMMARY ===")
    for k, r in sorted(existing.items()):
        lr_logo = r.get("lr_logo_mean", "N/A")
        mlp_logo = r.get("mlp_logo_mean", "N/A")
        gain_logo = r.get("gain_logo_pp", "N/A")
        log(f"  {k:30s} | LOGO LR={lr_logo} MLP={mlp_logo} gain={gain_logo}pp")

    log("=== Pretrained Unified Evaluation DONE ===")

if __name__ == "__main__":
    main()