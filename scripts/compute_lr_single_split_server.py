"""
Compute LR (single 80/20 split) for ALL from-scratch models.
Extracts embeddings for both codon and char tokenizers, then runs:
  - LR (single 80/20 split, random_state=42)
  - LR (5-fold CV, random_state=42)
  - MLP (single 80/20 split, random_state=42) [for verification]
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd, time, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import BertForMaskedLM, BertTokenizer
from collections import defaultdict

DATA_DIR = Path("./data")
MODEL_DIR = Path("./from_scratch_models")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
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
CODON_VOCAB = [c for c, aa in CODON_TABLE.items() if aa != '*']
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}

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

LOG_FILE = OUT_DIR / "lr_single_split.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

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
    log(f"  {len(cds)} CDS sequences")
    log("Loading ClinVar data...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    log(f"  {len(raw)} ClinVar records")
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
    log(f"  {len(var)} variants, {var['is_syn'].sum()} synonymous")
    return cds, var

def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        return pd.concat([df[df["label"] == 1].sample(n, random_state=42),
                          df[df["label"] == 0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df

def extract_embeddings(model_path, tokenizer_type, sequences, device="cuda:0", batch_size=8):
    log(f"  Loading model from {model_path}...")
    model = BertForMaskedLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    tok = BertTokenizer.from_pretrained(str(model_path))
    max_len = 512 if tokenizer_type == "codon" else 1536
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            encoded = tok(batch_seqs, padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded, output_hidden_states=True)
            hidden = output.hidden_states[-1]
            cls_emb = hidden[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)
            if (i + batch_size) % 200 == 0 or i + batch_size >= len(sequences):
                log(f"    Extracted {min(i + batch_size, len(sequences))}/{len(sequences)}")
    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embs)

def run_lr_single_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver='liblinear')
    lr.fit(X_train, y_train)
    return roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

def run_lr_5fold(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=2000, C=1.0, solver='liblinear')
        lr.fit(X_train, y_train)
        fold_aucs.append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
    return np.mean(fold_aucs), np.std(fold_aucs), fold_aucs

def run_mlp_single_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=42, early_stopping=True,
                        validation_fraction=0.1, solver='adam',
                        learning_rate_init=1e-3, alpha=1e-4)
    mlp.fit(X_train, y_train)
    return roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1])

def main():
    log("=== LR Single Split Computation START ===")
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    cds_cache, variants = load_data()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)
    df = balance(t3.copy(), max3)

    codon_seqs, char_seqs, labs = [], [], []
    for _, row in df.iterrows():
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        cs = build_codon_str(cds, cpos)
        chs = build_char_str(cds, cpos)
        if cs is None or chs is None: continue
        codon_seqs.append(cs)
        char_seqs.append(chs)
        labs.append(row["label"])

    labs = np.array(labs, dtype=int)
    log(f"SynPath: {len(labs)} variants, pos={labs.sum()}, neg={len(labs)-labs.sum()}")

    results_file = OUT_DIR / "from_scratch_lr_single_split_results.json"
    existing = {}
    if results_file.exists():
        existing = json.load(open(results_file))
        log(f"Loaded {len(existing)} existing results")

    for cond_name, model_subdir, tokenizer_type in CONDITIONS:
        if cond_name in existing:
            log(f"SKIP {cond_name} (already done)")
            continue

        model_path = MODEL_DIR / model_subdir
        if not model_path.exists():
            log(f"MISSING {cond_name}: {model_path}")
            continue

        sequences = codon_seqs if tokenizer_type == "codon" else char_seqs
        log(f"\n{'='*60}\n{cond_name} | tokenizer={tokenizer_type}\n{'='*60}")

        emb = extract_embeddings(model_path, tokenizer_type, sequences, device=device)
        n = min(len(emb), len(labs))
        emb = emb[:n]; y = labs[:n]
        log(f"  Embeddings: {emb.shape}")

        log("  Running LR (single 80/20 split)...")
        lr_8020 = run_lr_single_split(emb, y)
        log(f"  LR (80/20): {lr_8020:.4f}")

        log("  Running LR (5-fold CV)...")
        lr_5f_mean, lr_5f_std, lr_5f_folds = run_lr_5fold(emb, y)
        log(f"  LR (5-fold): {lr_5f_mean:.4f} +/- {lr_5f_std:.4f}")

        log("  Running MLP (single 80/20 split)...")
        mlp_8020 = run_mlp_single_split(emb, y)
        log(f"  MLP (80/20): {mlp_8020:.4f}")

        gain_8020 = (mlp_8020 - lr_8020) * 100
        gain_5f = (mlp_8020 - lr_5f_mean) * 100
        log(f"  Gain (LR8020→MLP8020): {gain_8020:+.1f} pp")
        log(f"  Gain (LR5f→MLP8020): {gain_5f:+.1f} pp")

        existing[cond_name] = {
            "tokenizer": tokenizer_type,
            "n": int(n),
            "lr_single_8020": round(float(lr_8020), 4),
            "lr_5fold_mean": round(float(lr_5f_mean), 4),
            "lr_5fold_std": round(float(lr_5f_std), 4),
            "lr_5fold_folds": [round(float(a), 4) for a in lr_5f_folds],
            "mlp_single_8020": round(float(mlp_8020), 4),
            "gain_lr8020_to_mlp8020_pp": round(float(gain_8020), 1),
            "gain_lr5f_to_mlp8020_pp": round(float(gain_5f), 1),
        }

        with open(results_file, "w") as f:
            json.dump(existing, f, indent=2)
        log(f"  Saved result for {cond_name}")

        del emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    log(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    for k, r in sorted(existing.items()):
        log(f"  {k:15s} | LR(80/20)={r['lr_single_8020']:.3f} | LR(5f)={r['lr_5fold_mean']:.3f} | "
            f"MLP(80/20)={r['mlp_single_8020']:.3f} | gain(80/20)={r['gain_lr8020_to_mlp8020_pp']:+.1f}pp | "
            f"gain(5f→80/20)={r['gain_lr5f_to_mlp8020_pp']:+.1f}pp")
    log("=== LR Single Split Computation DONE ===")

if __name__ == "__main__":
    main()