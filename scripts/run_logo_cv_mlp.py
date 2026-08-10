import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
def dna_to_rna(seq): return seq.replace("T", "U")
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)
def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos<1 or cpos>len(cds): return None
    ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
    c=dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
    cds=json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
    raw=pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",low_memory=False)
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
    return cds, var

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df["label"].sum()),int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n,random_state=42),df[df["label"]==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

def logo_cv_mlp(X, y, gene_ids, hidden_sizes=(128, 64), max_iter=100, random_state=42):
    unique_genes = sorted(set(gene_ids))
    fold_aucs = []
    for gene in unique_genes:
        test_mask = np.array([g == gene for g in gene_ids])
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 2:
            continue
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
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
    return np.array(fold_aucs)

def logo_cv_lr(X, y, gene_ids):
    unique_genes = sorted(set(gene_ids))
    fold_aucs = []
    for gene in unique_genes:
        test_mask = np.array([g == gene for g in gene_ids])
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 2:
            continue
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        lr.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        fold_aucs.append(auc)
    return np.array(fold_aucs)

EMB_FILES = {
    "codonbert": {"task3_synonymous": "codonbert_task3_synonymous_emb.npy"},
    "codonbert_hf": {"task3_synonymous": "codonbert_hf_task3_synonymous_emb.npy"},
    "encodon-80m": {"task3_synonymous": "encodon-80m_task3_synonymous_emb.npy"},
    "mrnabert": {"task3_synonymous": "mrnabert_task3_synonymous_emb.npy"},
    "ESM-2-650M": {"task3_synonymous": "ESM-2-650M_task3_synonymous_emb.npy"},
    "ESM-1b-650M": {"task3_synonymous": "ESM-1b-650M_task3_synonymous_emb.npy"},
}

def main():
    cds_cache, variants = load_data()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"]==1).sum())
    nb3 = int((t3["label"]==0).sum())
    max3 = min(np3*2, np3+nb3)

    df = balance(t3.copy(), max3)
    seqs, labs, gene_ids = [], [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
        if s is None: continue
        use_rna = False
        labs.append(row["label"])
        gene_ids.append(row["tx_id"])
    labs = np.array(labs, dtype=int)
    gene_ids = np.array(gene_ids)

    print(f"SynPath: {len(labs)} variants, {len(set(gene_ids))} genes")

    res_file = OUT_DIR / "logo_cv_mlp_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for model, tasks in EMB_FILES.items():
        for task, emb_file in tasks.items():
            if (model, task) in done_keys:
                print(f"  SKIP {model} {task}")
                continue
            emb_path = OUT_DIR / emb_file
            if not emb_path.exists():
                print(f"  MISSING {emb_path}")
                continue
            print(f"\n{'='*60}\n{model} | {task} (LOGO-CV)\n{'='*60}")
            emb = np.load(emb_path)
            n = min(len(emb), len(labs))
            emb = emb[:n]; y = labs[:n]; g = gene_ids[:n]
            print(f"  Embeddings: {emb.shape}, Genes: {len(set(g))}")

            print("  Running LOGO-CV LR...")
            lr_aucs = logo_cv_lr(emb, y, g)
            print(f"  LOGO-LR: {lr_aucs.mean():.4f} ± {lr_aucs.std():.4f} ({len(lr_aucs)} folds)")

            print("  Running LOGO-CV MLP...")
            mlp_aucs = logo_cv_mlp(emb, y, g)
            print(f"  LOGO-MLP: {mlp_aucs.mean():.4f} ± {mlp_aucs.std():.4f} ({len(mlp_aucs)} folds)")

            gain = mlp_aucs.mean() - lr_aucs.mean()
            print(f"  LOGO Gain: {gain:.4f} ({gain*100:.1f} pp)")

            r = {
                "model": model, "task": task, "n": int(n),
                "n_genes": int(len(set(g))),
                "n_valid_folds_lr": int(len(lr_aucs)),
                "n_valid_folds_mlp": int(len(mlp_aucs)),
                "logo_lr_mean": round(float(lr_aucs.mean()), 4),
                "logo_lr_std": round(float(lr_aucs.std()), 4),
                "logo_mlp_mean": round(float(mlp_aucs.mean()), 4),
                "logo_mlp_std": round(float(mlp_aucs.std()), 4),
                "logo_gain": round(float(gain), 4),
            }
            existing.append(r)
            with open(res_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nALL LOGO-CV RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:20s} | LOGO-LR={r['logo_lr_mean']:.4f}±{r['logo_lr_std']:.4f} | LOGO-MLP={r['logo_mlp_mean']:.4f}±{r['logo_mlp_std']:.4f} | Gain={r['logo_gain']:.4f}")

if __name__ == "__main__":
    main()