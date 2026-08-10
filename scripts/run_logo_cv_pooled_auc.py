import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd, time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16
LOG_FILE = OUT_DIR / "logo_cv_pooled_auc.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)
def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos<1 or cpos>len(cds): return None
    ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
    c=dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
    log("Loading CDS sequences...")
    cds=json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
    log(f"  Loaded {len(cds)} CDS sequences")
    log("Loading ClinVar data...")
    raw=pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",low_memory=False)
    log(f"  Loaded {len(raw)} ClinVar records")
    snv=raw[raw["Type"]=="single nucleotide variant"].copy()
    pk,bk=["Pathogenic","Likely pathogenic"],["Benign","Likely benign"]
    def classify(cs):
        cs=str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    snv["label"]=snv["ClinicalSignificance"].apply(classify)
    v=snv[snv["label"]>=0].copy()
    v=v[(v["ReferenceAlleleVCF"].str.len()==1)&(v["AlternateAlleleVCF"].str.len()==1)]
    p=v["Name"].apply(parse_hgvs)
    v["tx_id"]=p.apply(lambda x:x[0]); v["cpos"]=p.apply(lambda x:x[1])
    h=v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds.keys()))
    var=v[h].copy()
    var["is_syn"]=var["Name"].apply(lambda n:bool(re.search(r"p\.\w+\d+=",str(n))))
    log(f"  {len(var)} variants after filtering, {(var['is_syn']).sum()} synonymous")
    return cds, var

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df["label"].sum()),int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n,random_state=42),df[df["label"]==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

def precompute_gene_masks(gene_ids):
    gene_ids = np.array(gene_ids)
    unique_genes = sorted(set(gene_ids))
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    n = len(gene_ids)
    n_genes = len(unique_genes)
    gene_idx = np.array([gene_to_idx[g] for g in gene_ids])
    masks = np.zeros((n_genes, n), dtype=bool)
    for i in range(n):
        masks[gene_idx[i], i] = True
    return unique_genes, masks

def logo_cv_with_predictions(X, y, unique_genes, gene_masks, probe_type="lr",
                              hidden_sizes=(128, 64), max_iter=200, random_state=42):
    fold_aucs = []
    all_y_true = []
    all_y_proba = []
    fold_info = []
    total = len(unique_genes)
    t0 = time.time()
    for i, gene in enumerate(unique_genes):
        test_mask = gene_masks[i]
        train_mask = ~test_mask
        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_train < 10 or n_test < 2:
            continue
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        if probe_type == "lr":
            model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        else:
            model = MLPClassifier(hidden_layer_sizes=hidden_sizes, max_iter=max_iter,
                                  random_state=random_state, early_stopping=True,
                                  validation_fraction=0.1, solver='adam',
                                  learning_rate_init=1e-3, alpha=1e-4)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        fold_aucs.append(auc)
        all_y_true.extend(y_te.tolist())
        all_y_proba.extend(proba.tolist())
        fold_info.append({"gene": gene, "n_test": int(n_test), "auc": round(float(auc), 4)})
        if (i+1) % 20 == 0 or i == total - 1:
            elapsed = time.time() - t0
            log(f"    {probe_type.upper()} fold {i+1}/{total}, AUC={auc:.4f}, {len(fold_aucs)} valid folds, elapsed={elapsed:.0f}s")
    return np.array(fold_aucs), np.array(all_y_true), np.array(all_y_proba), fold_info

EMB_FILES = {
    "codonbert": {"task3_synonymous": "codonbert_task3_synonymous_emb.npy"},
    "codonbert_hf": {"task3_synonymous": "codonbert_hf_task3_synonymous_emb.npy"},
    "encodon-80m": {"task3_synonymous": "encodon-80m_task3_synonymous_emb.npy"},
    "mrnabert": {"task3_synonymous": "mrnabert_task3_synonymous_emb.npy"},
    "ESM-2-650M": {"task3_synonymous": "ESM-2-650M_task3_synonymous_emb.npy"},
    "ESM-1b-650M": {"task3_synonymous": "ESM-1b-650M_task3_synonymous_emb.npy"},
}

def main():
    log("=== LOGO-CV POOLED AUC START ===")
    cds_cache, variants = load_data()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"]==1).sum())
    nb3 = int((t3["label"]==0).sum())
    max3 = min(np3*2, np3+nb3)

    df = balance(t3.copy(), max3)
    labs, gene_ids = [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
        if s is None: continue
        labs.append(row["label"])
        gene_ids.append(row["tx_id"])
    labs = np.array(labs, dtype=int)
    gene_ids = np.array(gene_ids)

    log(f"SynPath: {len(labs)} variants, {len(set(gene_ids))} genes, pos={labs.sum()}, neg={len(labs)-labs.sum()}")

    log("Precomputing gene masks...")
    unique_genes, gene_masks = precompute_gene_masks(gene_ids)
    log(f"  {len(unique_genes)} unique genes, mask shape={gene_masks.shape}")

    results = []

    for model, tasks in EMB_FILES.items():
        for task, emb_file in tasks.items():
            log(f"\n{'='*60}\n{model} | {task} (LOGO-CV with predictions)\n{'='*60}")
            emb_path = OUT_DIR / emb_file
            if not emb_path.exists():
                log(f"  MISSING {emb_path}")
                continue
            emb = np.load(emb_path)
            n = min(len(emb), len(labs))
            emb = emb[:n]; y = labs[:n]
            cur_masks = gene_masks[:, :n]
            log(f"  Embeddings: {emb.shape}, Genes: {len(unique_genes)}")

            log("  Running LOGO-CV LR...")
            lr_aucs, lr_y_true, lr_y_proba, lr_fold_info = logo_cv_with_predictions(
                emb, y, unique_genes, cur_masks, probe_type="lr")
            lr_mean = float(lr_aucs.mean())
            lr_pooled = float(roc_auc_score(lr_y_true, lr_y_proba))
            log(f"  LOGO-LR: per-fold mean={lr_mean:.4f}, pooled={lr_pooled:.4f} ({len(lr_aucs)} folds)")

            log("  Running LOGO-CV MLP...")
            mlp_aucs, mlp_y_true, mlp_y_proba, mlp_fold_info = logo_cv_with_predictions(
                emb, y, unique_genes, cur_masks, probe_type="mlp")
            mlp_mean = float(mlp_aucs.mean())
            mlp_pooled = float(roc_auc_score(mlp_y_true, mlp_y_proba))
            log(f"  LOGO-MLP: per-fold mean={mlp_mean:.4f}, pooled={mlp_pooled:.4f} ({len(mlp_aucs)} folds)")

            gain_mean = mlp_mean - lr_mean
            gain_pooled = mlp_pooled - lr_pooled
            log(f"  LOGO Gain: per-fold mean={gain_mean:.4f} ({gain_mean*100:.1f} pp), pooled={gain_pooled:.4f} ({gain_pooled*100:.1f} pp)")

            r = {
                "model": model, "task": task, "n": int(n),
                "n_genes": int(len(unique_genes)),
                "n_valid_folds_lr": int(len(lr_aucs)),
                "n_valid_folds_mlp": int(len(mlp_aucs)),
                "logo_lr_mean": round(lr_mean, 4),
                "logo_lr_std": round(float(lr_aucs.std()), 4),
                "logo_lr_pooled": round(lr_pooled, 4),
                "logo_mlp_mean": round(mlp_mean, 4),
                "logo_mlp_std": round(float(mlp_aucs.std()), 4),
                "logo_mlp_pooled": round(mlp_pooled, 4),
                "logo_gain_per_fold_mean": round(gain_mean, 4),
                "logo_gain_pooled": round(gain_pooled, 4),
                "lr_fold_info": lr_fold_info,
                "mlp_fold_info": mlp_fold_info,
            }
            results.append(r)

            pred_out = OUT_DIR / f"logo_cv_predictions_{model}_{task}.npz"
            np.savez_compressed(pred_out,
                                lr_y_true=lr_y_true, lr_y_proba=lr_y_proba,
                                mlp_y_true=mlp_y_true, mlp_y_proba=mlp_y_proba)
            log(f"  Saved predictions to {pred_out}")

    res_file = OUT_DIR / "logo_cv_pooled_auc_results.json"
    with open(res_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n{'='*60}\nPOOLED AUC SUMMARY\n{'='*60}")
    log(f"{'Model':20s} | {'LR per-fold':>10s} | {'LR pooled':>10s} | {'MLP per-fold':>12s} | {'MLP pooled':>10s} | {'Gain per-fold':>13s} | {'Gain pooled':>12s}")
    log("-" * 100)
    for r in results:
        log(f"{r['model']:20s} | {r['logo_lr_mean']:10.4f} | {r['logo_lr_pooled']:10.4f} | {r['logo_mlp_mean']:12.4f} | {r['logo_mlp_pooled']:10.4f} | {r['logo_gain_per_fold_mean']:13.4f} | {r['logo_gain_pooled']:12.4f}")
    log("=== LOGO-CV POOLED AUC DONE ===")

if __name__ == "__main__":
    main()