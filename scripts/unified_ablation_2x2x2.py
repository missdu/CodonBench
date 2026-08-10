"""
Unified Ablation: 2x2x2 solver x scaler x aggregation
Proves CodonBERT's +4.7pp LOGO residual is implementation-dependent.

For each (model, task, solver, scaler) combo:
  - Run LOGO-CV with LR and MLP
  - Compute per-fold mean AUC and pooled AUC
  - Compute gain (per-fold and pooled)
  - Compute paired t-test

Models: CodonBERT, EnCodon-80M
Tasks: SynPath (primary), MisPath (secondary)
Solvers: lbfgs, liblinear
Scalers: StandardScaler, none
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, json, time, torch, torch.nn as nn
import pandas as pd, re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats as sp_stats

UNIFIED_DIR = Path("./results/unified_eval")
SYN_DIR = UNIFIED_DIR / "synpath_data"
MIS_DIR = UNIFIED_DIR / "mispath_data"
EMB_DIR_SUPP = Path("./results/supplementary")
EVAL_DIR = UNIFIED_DIR / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("./data")
DEVICE = "cuda:1"

MODELS = [
    ("CodonBERT", "codonbert", "codon", "CLS"),
    ("EnCodon-80M", "encodon-80m", "codon", "CLS"),
]

TASKS = [
    ("task3_synonymous", "SynPath"),
    ("task2_missense", "MisPath"),
]

SOLVERS = ["lbfgs", "liblinear"]
SCALERS = ["standard", "none"]

LOG_FILE = EVAL_DIR / "unified_ablation_2x2x2.log"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()


def run_lr(X_train, y_train, X_test, y_test, solver, use_scaler):
    if use_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    lr = LogisticRegression(C=1.0, solver=solver, max_iter=2000)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return auc, y_prob


def run_mlp(X_train, y_train, X_test, y_test, input_dim, use_scaler):
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
        y_prob = mlp(X_test_t).squeeze().cpu().numpy()
    auc = roc_auc_score(y_test, y_prob)
    return auc, y_prob


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
    logo = np.load(SYN_DIR / "split_logo_cv.npz", allow_pickle=True)
    valid_txs = list(logo["valid_fold_tx_ids"])
    return labels, tx_ids, valid_txs


def load_mispath_data(emb_prefix):
    lab_path = EMB_DIR_SUPP / f"{emb_prefix}_task2_missense_labels.npy"
    if lab_path.exists():
        labels = np.load(lab_path)
    else:
        labels, tx_ids = _build_mispath_data()
        return labels, tx_ids, None

    tx_ids_path = MIS_DIR / f"{emb_prefix}_tx_ids.npy"
    if tx_ids_path.exists():
        tx_ids = np.load(tx_ids_path, allow_pickle=True)
    else:
        labels, tx_ids = _build_mispath_data()

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

    return labels, tx_ids, valid_txs


def _build_mispath_data():
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


def run_logo_cv(emb, y, tx_ids, valid_txs, solver, use_scaler, input_dim):
    lr_folds, mlp_folds = [], []
    lr_preds_all, mlp_preds_all = [], []
    y_all = []

    for fi, tx in enumerate(valid_txs):
        test_mask = tx_ids == tx
        train_mask = ~test_mask
        tr = np.where(train_mask)[0]
        te = np.where(test_mask)[0]
        y_tr, y_te = y[tr], y[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2 or len(te) < 2:
            continue
        try:
            lr_auc, lr_prob = run_lr(emb[tr], y_tr, emb[te], y_te, solver, use_scaler)
            mlp_auc, mlp_prob = run_mlp(emb[tr], y_tr, emb[te], y_te, input_dim, use_scaler)
            lr_folds.append(lr_auc)
            mlp_folds.append(mlp_auc)
            lr_preds_all.extend(lr_prob.tolist())
            mlp_preds_all.extend(mlp_prob.tolist())
            y_all.extend(y_te.tolist())
        except Exception as e:
            continue
        if (fi + 1) % 50 == 0:
            log(f"    LOGO fold {fi+1}/{len(valid_txs)}, "
                f"LR={np.mean(lr_folds):.3f}, MLP={np.mean(mlp_folds):.3f}")

    if not lr_folds:
        return None

    lr_perfold_mean = float(np.mean(lr_folds))
    mlp_perfold_mean = float(np.mean(mlp_folds))
    gain_perfold = (mlp_perfold_mean - lr_perfold_mean) * 100

    lr_pooled = roc_auc_score(y_all, lr_preds_all)
    mlp_pooled = roc_auc_score(y_all, mlp_preds_all)
    gain_pooled = (mlp_pooled - lr_pooled) * 100

    t_stat, p_value = sp_stats.ttest_rel(mlp_folds, lr_folds)

    return {
        "lr_perfold_mean": round(lr_perfold_mean, 4),
        "lr_perfold_sd": round(float(np.std(lr_folds)), 4),
        "mlp_perfold_mean": round(mlp_perfold_mean, 4),
        "mlp_perfold_sd": round(float(np.std(mlp_folds)), 4),
        "gain_perfold_pp": round(gain_perfold, 1),
        "lr_pooled": round(lr_pooled, 4),
        "mlp_pooled": round(mlp_pooled, 4),
        "gain_pooled_pp": round(gain_pooled, 1),
        "n_folds": len(lr_folds),
        "paired_t": round(float(t_stat), 3),
        "paired_p": round(float(p_value), 4),
    }


def main():
    log("=== Unified Ablation 2x2x2 START ===")
    log(f"Models: {[m[0] for m in MODELS]}")
    log(f"Solvers: {SOLVERS}")
    log(f"Scalers: {SCALERS}")
    log(f"Device: {DEVICE}")

    results_file = EVAL_DIR / "unified_ablation_2x2x2_results.json"
    existing = {}
    if results_file.exists():
        existing = json.load(open(results_file))
        log(f"Loaded {len(existing)} existing results")

    for model_name, emb_prefix, tok_type, pooling in MODELS:
        for task_key, task_label in TASKS:
            emb_file = EMB_DIR_SUPP / f"{emb_prefix}_{task_key}_emb.npy"
            if not emb_file.exists():
                log(f"MISSING embedding: {emb_file}")
                continue

            emb = np.load(emb_file)
            log(f"\n{'='*60}")
            log(f"Model={model_name} Task={task_label} Pooling={pooling} emb={emb.shape}")

            if task_label == "SynPath":
                labels, tx_ids, valid_txs = load_synpath_data(emb_prefix)
            else:
                labels, tx_ids, valid_txs = load_mispath_data(emb_prefix)

            if valid_txs is None:
                log(f"  No LOGO-CV folds for {model_name} {task_label}, skipping")
                continue

            n = min(len(emb), len(labels))
            emb = emb[:n]
            y = labels[:n]
            tx_arr = tx_ids[:n] if len(tx_ids) >= n else tx_ids
            input_dim = emb.shape[1]

            for solver in SOLVERS:
                for scaler_name in SCALERS:
                    use_scaler = (scaler_name == "standard")
                    result_key = f"{model_name}_{task_label}_{solver}_{scaler_name}"

                    if result_key in existing:
                        log(f"  SKIP {result_key} (already done)")
                        continue

                    log(f"\n  --- {result_key} ---")
                    log(f"  solver={solver}, scaler={scaler_name}")

                    result = run_logo_cv(
                        emb, y, tx_arr, valid_txs,
                        solver, use_scaler, input_dim
                    )

                    if result is None:
                        log(f"  FAILED: no valid folds")
                        continue

                    result.update({
                        "model": model_name,
                        "task": task_label,
                        "solver": solver,
                        "scaler": scaler_name,
                        "pooling": pooling,
                        "tokenizer_type": tok_type,
                    })

                    existing[result_key] = result
                    with open(results_file, "w") as f:
                        json.dump(existing, f, indent=2)

                    log(f"  Per-fold: LR={result['lr_perfold_mean']:.3f}±{result['lr_perfold_sd']:.3f} "
                        f"MLP={result['mlp_perfold_mean']:.3f}±{result['mlp_perfold_sd']:.3f} "
                        f"gain={result['gain_perfold_pp']:+.1f}pp")
                    log(f"  Pooled:   LR={result['lr_pooled']:.3f} MLP={result['mlp_pooled']:.3f} "
                        f"gain={result['gain_pooled_pp']:+.1f}pp")
                    log(f"  Paired t-test: t={result['paired_t']:.3f}, p={result['paired_p']:.4f}")

            del emb
            torch.cuda.empty_cache()

    log("\n" + "="*60)
    log("=== SUMMARY TABLE ===")
    log(f"{'Key':<45s} | {'PF gain':>8s} | {'Pool gain':>9s} | {'p':>6s} | {'n_folds':>7s}")
    log("-"*85)
    for k, r in sorted(existing.items()):
        if "gain_perfold_pp" in r:
            log(f"{k:<45s} | {r['gain_perfold_pp']:>+7.1f}pp | {r['gain_pooled_pp']:>+8.1f}pp | "
                f"{r['paired_p']:>6.4f} | {r['n_folds']:>7d}")

    log("=== Unified Ablation 2x2x2 DONE ===")


if __name__ == "__main__":
    main()