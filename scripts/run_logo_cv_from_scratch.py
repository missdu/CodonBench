import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd, time, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import BertForMaskedLM, BertTokenizer

DATA_DIR = Path("./data")
MODEL_DIR = Path("./from_scratch_models")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16
LOG_FILE = OUT_DIR / "logo_cv_from_scratch.log"

CONDITIONS = [
    ("codon-v1", "codon-bert-ablation", "codon"),
    ("char-v1", "char-bert-ablation", "char"),
    ("codon-v3a", "codon-bert-ablation-v3a-real100ep", "codon"),
    ("char-v3a", "char-bert-ablation-v3a-real100ep", "char"),
    ("codon-v3b", "codon-bert-ablation-v3b-ensembl", "codon"),
    ("char-v3b", "char-bert-ablation-v3b-ensembl", "char"),
    ("codon-v4", "codon-bert-ablation-v4-scale110m", "codon"),
    ("char-v4", "char-bert-ablation-v4-scale110m", "char"),
]

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def build_char_str(cds, cpos, ctx=CODON_CONTEXT * 3):
    if cpos < 1 or cpos > len(cds): return None
    start = max(0, cpos - 1 - ctx)
    end = min(len(cds), cpos + ctx)
    return cds[start:end]

def load_data():
    log("Loading CDS sequences...")
    cds = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    log(f"  Loaded {len(cds)} CDS sequences")
    log("Loading ClinVar data...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    log(f"  Loaded {len(raw)} ClinVar records")
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
    log(f"  {len(var)} variants after filtering, {(var['is_syn']).sum()} synonymous")
    return cds, var

def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        return pd.concat([df[df["label"] == 1].sample(n, random_state=42),
                          df[df["label"] == 0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
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

def logo_cv_lr(X, y, unique_genes, gene_masks):
    fold_aucs = []
    total = len(unique_genes)
    t0 = time.time()
    for i, gene in enumerate(unique_genes):
        test_mask = gene_masks[i]
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 2: continue
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2: continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        fold_aucs.append(auc)
        if (i + 1) % 50 == 0 or i == total - 1:
            log(f"    LR fold {i+1}/{total}, AUC={auc:.4f}, {len(fold_aucs)} valid, {time.time()-t0:.0f}s")
    return np.array(fold_aucs)

def logo_cv_mlp(X, y, unique_genes, gene_masks):
    fold_aucs = []
    total = len(unique_genes)
    t0 = time.time()
    for i, gene in enumerate(unique_genes):
        test_mask = gene_masks[i]
        train_mask = ~test_mask
        if train_mask.sum() < 10 or test_mask.sum() < 2: continue
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2: continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                            random_state=42, early_stopping=True,
                            validation_fraction=0.1, solver='adam',
                            learning_rate_init=1e-3, alpha=1e-4)
        mlp.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
        fold_aucs.append(auc)
        if (i + 1) % 20 == 0 or i == total - 1:
            log(f"    MLP fold {i+1}/{total}, AUC={auc:.4f}, {len(fold_aucs)} valid, {time.time()-t0:.0f}s")
    return np.array(fold_aucs)

def extract_embeddings(model_path, tokenizer_type, sequences, batch_size=8):
    log(f"  Loading model from {model_path}...")
    model = BertForMaskedLM.from_pretrained(str(model_path))
    model.eval()
    tok = BertTokenizer.from_pretrained(str(model_path))
    max_len = 512 if tokenizer_type == "codon" else 1536
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            encoded = tok(batch_seqs, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
            output = model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            cls_emb = hidden[:, 0, :].numpy()
            all_embs.append(cls_emb)
            if (i + batch_size) % 200 == 0 or i + batch_size >= len(sequences):
                log(f"    Extracted {min(i + batch_size, len(sequences))}/{len(sequences)} embeddings")
    return np.vstack(all_embs)

def main():
    log("=== LOGO-CV From-Scratch START ===")
    cds_cache, variants = load_data()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)
    df = balance(t3.copy(), max3)

    codon_seqs, char_seqs, labs, gene_ids = [], [], [], []
    for _, row in df.iterrows():
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        cs = build_codon_str(cds, cpos)
        chs = build_char_str(cds, cpos)
        if cs is None or chs is None: continue
        codon_seqs.append(cs)
        char_seqs.append(chs)
        labs.append(row["label"])
        gene_ids.append(row["tx_id"])

    labs = np.array(labs, dtype=int)
    gene_ids = np.array(gene_ids)
    log(f"SynPath: {len(labs)} variants, {len(set(gene_ids))} genes, pos={labs.sum()}, neg={len(labs)-labs.sum()}")

    log("Precomputing gene masks...")
    unique_genes, gene_masks = precompute_gene_masks(gene_ids)
    log(f"  {len(unique_genes)} unique genes, mask shape={gene_masks.shape}")

    res_file = OUT_DIR / "logo_cv_from_scratch_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {r["model"] for r in existing}
    log(f"Already done: {done_keys}")

    for cond_name, model_subdir, tokenizer_type in CONDITIONS:
        if cond_name in done_keys:
            log(f"  SKIP {cond_name}")
            continue
        model_path = MODEL_DIR / model_subdir
        if not model_path.exists():
            log(f"  MISSING model path: {model_path}")
            continue
        sequences = codon_seqs if tokenizer_type == "codon" else char_seqs
        log(f"\n{'='*60}\n{cond_name} | tokenizer={tokenizer_type} (LOGO-CV)\n{'='*60}")

        emb = extract_embeddings(model_path, tokenizer_type, sequences, batch_size=8)
        n = min(len(emb), len(labs))
        emb = emb[:n]; y = labs[:n]
        cur_masks = gene_masks[:, :n]
        log(f"  Embeddings: {emb.shape}, Genes: {len(unique_genes)}")

        log("  Running LOGO-CV LR...")
        lr_aucs = logo_cv_lr(emb, y, unique_genes, cur_masks)
        log(f"  LOGO-LR: {lr_aucs.mean():.4f} +/- {lr_aucs.std():.4f} ({len(lr_aucs)} folds)")

        log("  Running LOGO-CV MLP...")
        mlp_aucs = logo_cv_mlp(emb, y, unique_genes, cur_masks)
        log(f"  LOGO-MLP: {mlp_aucs.mean():.4f} +/- {mlp_aucs.std():.4f} ({len(mlp_aucs)} folds)")

        gain = mlp_aucs.mean() - lr_aucs.mean()
        log(f"  LOGO Gain: {gain:.4f} ({gain*100:.1f} pp)")

        r = {
            "model": cond_name, "tokenizer": tokenizer_type,
            "n": int(n), "n_genes": int(len(unique_genes)),
            "n_valid_folds_lr": int(len(lr_aucs)),
            "n_valid_folds_mlp": int(len(mlp_aucs)),
            "logo_lr_mean": round(float(lr_aucs.mean()), 4),
            "logo_lr_std": round(float(lr_aucs.std()), 4),
            "logo_mlp_mean": round(float(mlp_aucs.mean()), 4),
            "logo_mlp_std": round(float(mlp_aucs.std()), 4),
            "logo_gain_pp": round(float(gain * 100), 1),
        }
        existing.append(r)
        with open(res_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        log(f"  Saved result for {cond_name}")

        del emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\n{'='*60}\nALL LOGO-CV FROM-SCRATCH RESULTS\n{'='*60}")
    for r in existing:
        log(f"  {r['model']:15s} | tok={r['tokenizer']:5s} | LOGO-LR={r['logo_lr_mean']:.4f} | LOGO-MLP={r['logo_mlp_mean']:.4f} | Gain={r['logo_gain_pp']:.1f} pp")
    log("=== LOGO-CV From-Scratch DONE ===")

if __name__ == "__main__":
    main()
