"""
Compute LR (single 80/20 split) for ALL from-scratch models.
Uses custom CodonTokenizer/CharTokenizer (same as run_from_scratch_ablation.py).
"""
import sys; sys.path.insert(0, ".")
import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import numpy as np, json, re, pandas as pd, time, torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from transformers import BertForMaskedLM, PreTrainedTokenizer

DATA_DIR = Path("./data")
MODEL_DIR = Path("./from_scratch_models")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODON_VOCAB = [
    "TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG",
    "TAT", "TAC", "TAA", "TAG", "TGT", "TGC", "TGA", "TGG",
    "CTT", "CTC", "CTA", "CTG", "CCT", "CCC", "CCA", "CCG",
    "CAT", "CAC", "CAA", "CAG", "CGT", "CGC", "CGA", "CGG",
    "ATT", "ATC", "ATA", "ATG", "ACT", "ACC", "ACA", "ACG",
    "AAT", "AAC", "AAA", "AAG", "AGT", "AGC", "AGA", "AGG",
    "GTT", "GTC", "GTA", "GTG", "GCT", "GCC", "GCA", "GCG",
    "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA", "GGG",
]
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

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

LOG_FILE = OUT_DIR / "lr_single_split_v2.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()

class CodonTokenizer(PreTrainedTokenizer):
    vocab_size = len(SPECIAL_TOKENS) + len(CODON_VOCAB)
    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 512)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        super().__init__(**kwargs)
    def _tokenize(self, text):
        text = text.upper().strip()
        if " " in text:
            return text.split()
        return [text[i:i+3] for i in range(0, len(text) - 2, 3)]
    def _convert_token_to_id(self, token):
        return self.vocab_dict.get(token, self.vocab_dict["[UNK]"])
    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "[UNK]")
    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)
    def get_vocab(self):
        return dict(self.vocab_dict)
    @property
    def pad_token_id(self):
        return self.vocab_dict["[PAD]"]
    @property
    def cls_token_id(self):
        return self.vocab_dict["[CLS]"]
    @property
    def sep_token_id(self):
        return self.vocab_dict["[SEP]"]
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep
    def save_vocabulary(self, save_dir, filename_prefix=None):
        vocab_file = os.path.join(save_dir, (filename_prefix or "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            for token, idx in sorted(self.vocab_dict.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        return (vocab_file,)

class CharTokenizer(PreTrainedTokenizer):
    CHAR_VOCAB = ["A", "T", "G", "C"]
    vocab_size = len(SPECIAL_TOKENS) + len(CHAR_VOCAB)
    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + self.CHAR_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 1536)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        super().__init__(**kwargs)
    def _tokenize(self, text):
        text = text.upper().strip().replace(" ", "")
        return list(text)
    def _convert_token_to_id(self, token):
        return self.vocab_dict.get(token, self.vocab_dict["[UNK]"])
    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "[UNK]")
    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    def get_vocab(self):
        return dict(self.vocab_dict)
    @property
    def pad_token_id(self):
        return self.vocab_dict["[PAD]"]
    @property
    def cls_token_id(self):
        return self.vocab_dict["[CLS]"]
    @property
    def sep_token_id(self):
        return self.vocab_dict["[SEP]"]
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep
    def save_vocabulary(self, save_dir, filename_prefix=None):
        vocab_file = os.path.join(save_dir, (filename_prefix or "") + "vocab.txt")
        with open(vocab_file, "w") as f:
            for token, idx in sorted(self.vocab_dict.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
        return (vocab_file,)

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

def tokenize_batch(tokenizer, sequences, max_length):
    all_ids = []
    for s in sequences:
        tokens = tokenizer._tokenize(s)
        ids = [tokenizer.cls_token_id] + [tokenizer._convert_token_to_id(t) for t in tokens[:max_length - 2]] + [tokenizer.sep_token_id]
        all_ids.append(ids)
    max_l = min(max(len(ids) for ids in all_ids), max_length)
    padded = [ids[:max_l] + [tokenizer.pad_token_id] * (max_l - len(ids[:max_l])) for ids in all_ids]
    return torch.tensor(padded, dtype=torch.long)

def extract_embeddings(model_path, tokenizer, sequences, device="cuda:1", batch_size=8):
    log(f"  Loading model from {model_path}...")
    model = BertForMaskedLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    max_len = tokenizer.model_max_length
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            input_ids = tokenize_batch(tokenizer, batch_seqs, max_len).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-2]  # layer=-2, same as run_from_scratch_ablation.py
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)  # mean pooling
            all_embs.append(embeddings.cpu().numpy())
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
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
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
        lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        lr.fit(X_train, y_train)
        fold_aucs.append(roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
    return np.mean(fold_aucs), np.std(fold_aucs), fold_aucs

def run_mlp_single_split(X, y):
    import torch.nn as nn
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    mlp = nn.Sequential(
        nn.Linear(X_train.shape[1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    ).to(device)
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

def main():
    log("=== LR Single Split v2 START ===")

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

        tokenizer = CodonTokenizer() if tokenizer_type == "codon" else CharTokenizer()
        sequences = codon_seqs if tokenizer_type == "codon" else char_seqs

        log(f"\n{'='*60}\n{cond_name} | tokenizer={tokenizer_type}\n{'='*60}")

        emb = extract_embeddings(model_path, tokenizer, sequences)
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
        log(f"  Gain (LR8020->MLP8020): {gain_8020:+.1f} pp")
        log(f"  Gain (LR5f->MLP8020): {gain_5f:+.1f} pp")

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
        torch.cuda.empty_cache()

    log(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    for k, r in sorted(existing.items()):
        log(f"  {k:15s} | LR(80/20)={r['lr_single_8020']:.3f} | LR(5f)={r['lr_5fold_mean']:.3f} | "
            f"MLP(80/20)={r['mlp_single_8020']:.3f} | gain(80/20)={r['gain_lr8020_to_mlp8020_pp']:+.1f}pp | "
            f"gain(5f->80/20)={r['gain_lr5f_to_mlp8020_pp']:+.1f}pp")
    log("=== LR Single Split v2 DONE ===")

if __name__ == "__main__":
    main()