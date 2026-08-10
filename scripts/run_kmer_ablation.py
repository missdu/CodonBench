"""P1-1: k-mer tokenization ablation.

Train 5 BERT models with different tokenizers on the same corpus:
  1. 3-mer-aligned (codon tokenizer, from position 0)
  2. 3-mer-offset-1 (from position 1, not aligned to codon boundaries)
  3. 3-mer-offset-2 (from position 2, not aligned to codon boundaries)
  4. 4-mer (4-nt tokens, not aligned to codon boundaries)
  5. 6-mer-aligned (6-nt tokens, aligned to codon pair boundaries)

All models use the same BERT architecture (~20M params), same corpus
(v3b: 114K Ensembl CDS), same training params (30 epochs).

Key prediction: if inductive bias hypothesis is correct,
  SynPath MLP: 3-mer-aligned > 6-mer > 3-mer-offset > 4-mer > char

This tests whether the advantage comes from:
  (a) 3-nt grouping size (any 3-mer works) → 3-mer-aligned ≈ 3-mer-offset
  (b) Codon boundary alignment (only aligned 3-mer works) → 3-mer-aligned >> 3-mer-offset
"""
import sys, os, json, re, time, random
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/kmer_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("./kmer_ablation_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

TOKENIZER_CONFIGS = {
    "3mer_aligned": {"k": 3, "offset": 0, "desc": "Codon-aligned 3-mer (=codon tokenizer)"},
    "3mer_offset1": {"k": 3, "offset": 1, "desc": "Offset-1 3-mer (not codon-aligned)"},
    "3mer_offset2": {"k": 3, "offset": 2, "desc": "Offset-2 3-mer (not codon-aligned)"},
    "4mer": {"k": 4, "offset": 0, "desc": "4-mer (not codon-aligned)"},
    "6mer_aligned": {"k": 6, "offset": 0, "desc": "6-mer aligned to codon pairs"},
}

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

def tokenize_sequence(seq, k, offset=0):
    """Tokenize a DNA sequence into k-mers with given offset."""
    if offset > 0:
        seq = seq[offset:]
    tokens = [seq[i:i+k] for i in range(0, len(seq) - (len(seq) - offset) % k, k)]
    tokens = [t for t in tokens if len(t) == k]
    return " ".join(tokens)

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_tokenized_context(cds, cpos, tokenizer_name, ctx=CODON_CONTEXT):
    """Build tokenized context around variant position for given tokenizer."""
    if cpos < 1 or cpos > len(cds): return None
    k = TOKENIZER_CONFIGS[tokenizer_name]["k"]
    offset = TOKENIZER_CONFIGS[tokenizer_name]["offset"]

    codon_idx = (cpos - 1) // 3
    start = max(0, codon_idx - ctx)
    end = min(len(cds) // 3, codon_idx + ctx + 1)
    subseq = cds[start*3:end*3]
    if not subseq: return None
    return tokenize_sequence(subseq, k, offset)

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

def build_vocab(tokenizer_name):
    """Build vocabulary for given tokenizer from all possible k-mers."""
    k = TOKENIZER_CONFIGS[tokenizer_name]["k"]
    bases = "ACGT"
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    idx = 5
    from itertools import product
    for combo in product(bases, repeat=k):
        token = "".join(combo)
        vocab[token] = idx
        idx += 1
    return vocab

def pretrain_model(tokenizer_name, corpus_path, epochs=30, batch_size=32, lr=5e-5):
    """Pretrain a BERT model with given tokenizer on CDS corpus."""
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer
    from torch.utils.data import Dataset, DataLoader

    print(f"\n{'='*60}")
    print(f"PRETRAINING: {tokenizer_name}")
    print(f"{'='*60}")

    vocab = build_vocab(tokenizer_name)
    k = TOKENIZER_CONFIGS[tokenizer_name]["k"]
    offset = TOKENIZER_CONFIGS[tokenizer_name]["offset"]

    # Save vocab as tokenizer
    vocab_path = MODEL_DIR / f"vocab_{tokenizer_name}.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    tokenizer = BertTokenizer(str(vocab_path), do_lower_case=False)
    print(f"  Vocab size: {len(vocab)}")

    # Load corpus
    with open(corpus_path) as f:
        cds_data = json.load(f)
    sequences = list(cds_data.values())
    print(f"  Corpus: {len(sequences)} sequences")

    # Tokenize
    tokenized = []
    for seq in sequences[:1000]:  # Start with subset for testing
        tokens = tokenize_sequence(seq, k, offset)
        if tokens:
            tokenized.append(tokens)
    print(f"  Tokenized: {len(tokenized)} sequences")

    # Configure model
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
    )
    model = BertForMaskedLM(config)
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Save model path
    model_path = MODEL_DIR / f"bert-{tokenizer_name}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"  Model saved to {model_path}")

    return model_path

def evaluate_model(tokenizer_name, task_name, task_df, max_n, cds_cache):
    """Evaluate a pretrained k-mer model on a task."""
    from transformers import BertForMaskedLM, BertTokenizer

    model_path = MODEL_DIR / f"bert-{tokenizer_name}"
    if not model_path.exists():
        print(f"  Model not found at {model_path}, skipping evaluation")
        return None

    emb_path = OUT_DIR / f"{tokenizer_name}_{task_name}_emb.npy"
    lab_path = OUT_DIR / f"{tokenizer_name}_{task_name}_labels.npy"

    if emb_path.exists() and lab_path.exists():
        print(f"  Loading cached embeddings from {emb_path}")
        emb = np.load(emb_path)
        labs = np.load(lab_path)
    else:
        print(f"  Loading model from {model_path}...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForMaskedLM.from_pretrained(model_path)
        model = model.bert.to(DEVICE).eval()

        df = balance(task_df.copy(), max_n)
        seqs, labs_list = [], []
        for _, row in df.iterrows():
            s = build_tokenized_context(cds_cache.get(row["tx_id"], ""), int(row["cpos"]), tokenizer_name)
            if s is None: continue
            seqs.append(s)
            labs_list.append(row["label"])
        labs = np.array(labs_list, dtype=int)
        print(f"  {task_name}: {len(seqs)} sequences")

        # Extract embeddings
        all_embs = []
        bs = 16
        for i in range(0, len(seqs), bs):
            encoded = tokenizer(seqs[i:i+bs], padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = model(**{k: v.to(DEVICE) for k, v in encoded.items()})
            all_embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
            if (i // bs) % 50 == 0:
                print(f"  Batch {i//bs+1}/{(len(seqs)-1)//bs+1}")
        emb = np.vstack(all_embs)
        np.save(emb_path, emb)
        np.save(lab_path, labs)
        del model
        torch.cuda.empty_cache()

    print(f"  Embeddings shape: {emb.shape}, Labels: {len(labs)}")

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
        "tokenizer": tokenizer_name,
        "task": task_name,
        "k": TOKENIZER_CONFIGS[tokenizer_name]["k"],
        "offset": TOKENIZER_CONFIGS[tokenizer_name]["offset"],
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    tokenizer_key = sys.argv[2] if len(sys.argv) > 2 else "all"

    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    corpus_path = DATA_DIR / "ensembl_cds" / "ensembl_human_cds.json"

    tokenizers = list(TOKENIZER_CONFIGS.keys())
    if tokenizer_key != "all":
        tokenizers = [tokenizer_key]

    if mode == "pretrain":
        for tok_name in tokenizers:
            pretrain_model(tok_name, corpus_path)

    elif mode == "eval":
        tasks = [
            ("task2_missense", t2, 5000),
            ("task3_synonymous", t3, max3),
        ]

        res_file = OUT_DIR / "kmer_ablation_results.json"
        existing = []
        if res_file.exists():
            existing = json.load(open(res_file))
        done_keys = {(r["tokenizer"], r["task"]) for r in existing}

        for tok_name in tokenizers:
            for task_name, task_df, max_n in tasks:
                if (tok_name, task_name) in done_keys:
                    print(f"  SKIP {tok_name} {task_name} (already done)")
                    continue
                print(f"\n{'='*60}\n{tok_name} | {task_name}\n{'='*60}")
                r = evaluate_model(tok_name, task_name, task_df, max_n, cds_cache)
                if r:
                    existing.append(r)
                    with open(res_file, "w") as f:
                        json.dump(existing, f, indent=2, default=str)

        print(f"\n{'='*60}\nk-MER ABLATION RESULTS\n{'='*60}")
        for r in existing:
            print(f"  {r['tokenizer']:20s} | {r['task']:25s} | "
                  f"CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
                  f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")

    elif mode == "all":
        for tok_name in tokenizers:
            pretrain_model(tok_name, corpus_path)
        for tok_name in tokenizers:
            for task_name, task_df, max_n in tasks:
                evaluate_model(tok_name, task_name, task_df, max_n, cds_cache)

if __name__ == "__main__":
    main()