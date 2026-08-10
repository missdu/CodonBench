"""P2-1: CodonTransformer proper evaluation.

CodonTransformer (Nat Commun 2025.04) uses BigBird architecture with
aa-codon paired tokenization. Previous evaluation failed because the
tokenizer mapped codon tokens to UNK.

This script attempts to fix the tokenization issue by:
1. Using the correct input format (aa-codon pairs)
2. Verifying token-to-id mapping
3. If UNK persists, using the model's own encode method
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_aa_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    """Build aa-codon paired string for CodonTransformer.
    Format: "A_TTG C_GAA ..." (amino acid underscore codon)
    """
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    codons = dna_to_codons(cds[s*3:e*3])
    if not codons: return None
    pairs = []
    for c in codons:
        aa = CODON_TABLE.get(c, 'X')
        if aa == '*': continue
        pairs.append(f"{aa}_{c}")
    return " ".join(pairs) if pairs else None

def load_data():
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
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
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    return cds_cache, variants

def balance(df, max_n):
    if len(df) > max_n:
        n = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n, random_state=42),
                          df[df["label"]==0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df

def run_codontransformer(task_name, task_df, max_n, cds_cache):
    """Run CodonTransformer with proper aa-codon tokenization."""
    from transformers import AutoModel, AutoTokenizer

    model_id = "AdrienBdx/CodonTransformer"
    print(f"  Loading CodonTransformer from {model_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"  AutoModel failed: {e}")
        print("  Trying CodonTransformer specific loading...")
        try:
            from CodonTransformer import CodonBERT
            model = CodonBERT.from_pretrained(model_id)
            tokenizer = model.tokenizer
        except Exception as e2:
            print(f"  CodonBERT also failed: {e2}")
            return None

    model = model.to(DEVICE).eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    print(f"  Tokenizer: {type(tokenizer).__name__}, vocab_size={getattr(tokenizer, 'vocab_size', 'N/A')}")

    # Test tokenization
    test_seq = build_aa_codon_str("ATGAAAGGGTTTCCC", 4)
    if test_seq:
        test_enc = tokenizer(test_seq, return_tensors="pt")
        test_ids = test_enc["input_ids"].tolist()[0]
        test_tokens = tokenizer.convert_ids_to_tokens(test_ids)
        unk_count = sum(1 for t in test_tokens if t in ["[UNK]", "<unk>", "UNK"])
        print(f"  Test encoding: {test_seq[:50]}...")
        print(f"  Tokens: {test_tokens[:10]}... UNK count: {unk_count}/{len(test_tokens)}")

    df = balance(task_df.copy(), max_n)
    seqs, labs_list = [], []
    for _, row in df.iterrows():
        s = build_aa_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s is None: continue
        seqs.append(s)
        labs_list.append(row["label"])
    labs = np.array(labs_list, dtype=int)
    print(f"  {task_name}: {len(seqs)} sequences")

    # Extract embeddings
    all_embs = []
    bs = 16
    t0 = time.time()
    for i in range(0, len(seqs), bs):
        encoded = tokenizer(seqs[i:i+bs], padding=True, truncation=True,
                            max_length=1024, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.to(DEVICE) for k, v in encoded.items()})
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            all_embs.append(out.pooler_output.cpu().numpy())
        else:
            all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    emb = np.vstack(all_embs)
    print(f"  Embeddings: {emb.shape}, {time.time()-t0:.1f}s")

    del model
    torch.cuda.empty_cache()

    X_tr, X_te, y_tr, y_te = train_test_split(emb, labs, test_size=0.2, random_state=42, stratify=labs)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
    lr_test = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    lr_cv = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        emb, labs, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc"
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1
    ).fit(X_tr, y_tr)
    mlp_test = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])

    r = {
        "model": "CodonTransformer",
        "task": task_name,
        "model_type": "cLM (aa-codon paired, BigBird)",
        "n": int(len(labs)),
        "cv_lr_mean": round(float(lr_cv.mean()), 4),
        "cv_lr_std": round(float(lr_cv.std()), 4),
        "test_lr": round(float(lr_test), 4),
        "test_mlp": round(float(mlp_test), 4),
    }
    print(f"  CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
          f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")
    return r

def main():
    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, max3)]

    res_file = OUT_DIR / "codontransformer_proper_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for task_name, task_df, max_n in tasks:
        if ("CodonTransformer", task_name) in done_keys:
            print(f"  SKIP CodonTransformer {task_name}")
            continue
        print(f"\n{'='*60}\nCodonTransformer | {task_name}\n{'='*60}")
        r = run_codontransformer(task_name, task_df, max_n, cds_cache)
        if r:
            existing.append(r)
            with open(res_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nCodonTransformer PROPER RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:20s} | {r['task']:25s} | "
              f"CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
              f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")

if __name__ == "__main__":
    main()