"""
From-scratch BERT pretraining ablation for CodonBench v14.

Trains two BERT models from scratch with IDENTICAL architecture and corpus,
differing ONLY in tokenizer:
  1. codon-bert-ablation: codon-level tokenizer (vocab=69, space-separated codons)
  2. char-bert-ablation: character-level tokenizer (vocab=5, A/T/G/C + special)

This is the causal test for "tokenization determines access to I(σ;Y|A)":
  - If codon-tokenized beats char-tokenized on synonymous variants (Task 2),
    with architecture and corpus matched, the advantage is causally attributable
    to tokenization, not to pretraining corpus, scale, or warm-start.

Usage:
  python run_from_scratch_ablation.py --mode pretrain --tokenizer_type codon
  python run_from_scratch_ablation.py --mode pretrain --tokenizer_type char
  python run_from_scratch_ablation.py --mode eval --tokenizer_type codon
  python run_from_scratch_ablation.py --mode eval --tokenizer_type char
  python run_from_scratch_ablation.py --mode all
"""

import argparse
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

RESULT_DIR = os.path.expanduser("./results")
PRETRAIN_DIR = os.path.expanduser("./from_scratch_models")
CDS_DATA_DIR = os.path.expanduser("./data/cds_sequences")


# ============================================================
# Tokenizers
# ============================================================

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


class CodonTokenizer(PreTrainedTokenizer):
    """Codon-level tokenizer: splits CDS into space-separated 3-mers."""

    vocab_size = len(SPECIAL_TOKENS) + len(CODON_VOCAB)  # 5 + 64 = 69

    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 512)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", SPECIAL_TOKENS[4])
        super().__init__(**kwargs)

    def _tokenize(self, text):
        text = text.upper().strip()
        if " " in text:
            codons = text.split()
        else:
            codons = [text[i:i+3] for i in range(0, len(text) - 2, 3)]
        return codons

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
    def mask_token_id(self):
        return self.vocab_dict[SPECIAL_TOKENS[4]]

    @property
    def unk_token_id(self):
        return self.vocab_dict["[UNK]"]

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
    """Character-level tokenizer: splits CDS into individual nucleotides."""

    CHAR_VOCAB = ["A", "T", "G", "C"]
    vocab_size = len(SPECIAL_TOKENS) + len(CHAR_VOCAB)  # 5 + 4 = 9

    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + self.CHAR_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 1536)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", SPECIAL_TOKENS[4])
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
    def mask_token_id(self):
        return self.vocab_dict[SPECIAL_TOKENS[4]]

    @property
    def unk_token_id(self):
        return self.vocab_dict["[UNK]"]

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


# ============================================================
# Dataset
# ============================================================

class CDSDataset(Dataset):
    """CDS sequences for MLM pretraining."""

    def __init__(self, sequences, tokenizer, max_length):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": [1] * len(encoding["input_ids"]),
        }


class DataCollatorForCDS:
    """Custom collator that pads to max length in batch."""

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        input_ids = []
        attention_mask = []
        labels = []

        for e in examples:
            ids = e["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(attention_mask == 0, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# Data loading
# ============================================================

def load_cds_sequences(data_source="expanded"):
    """Load CDS sequences for pretraining.
    
    data_source options:
      - "expanded": 55K CDS (4342 CaLM + 500 ClinVar + 50658 synthetic)
      - "real_only": 4842 real CDS only (4342 CaLM + 500 ClinVar)
      - "ensembl": Ensembl human CDS (downloaded separately)
    """
    if data_source == "ensembl":
        ensembl_file = os.path.expanduser(
            "./data/ensembl_cds/ensembl_human_cds.json"
        )
        if os.path.exists(ensembl_file):
            with open(ensembl_file) as f:
                data = json.load(f)
            sequences = data["sequences"]
            print(f"Loaded {len(sequences)} CDS sequences from Ensembl corpus")
            return sequences
        else:
            print(f"ERROR: Ensembl CDS file not found at {ensembl_file}")
            print("Run scripts/download_ensembl_cds.py first.")
            sys.exit(1)

    if data_source == "real_only":
        sequences = []
        clinvar_dir = os.path.expanduser("./data/task2_clinvar")
        cds_cache = os.path.join(clinvar_dir, "cds_sequences.json")
        if os.path.exists(cds_cache):
            with open(cds_cache) as f:
                data = json.load(f)
            for tid, seq in data.items():
                if isinstance(seq, str) and len(seq) >= 30:
                    sequences.append(seq)
            print(f"After ClinVar cache: {len(sequences)} CDS sequences")

        calm_species_dir = "<MODEL_PATH>/cLMs/CaLM/data/species"
        if os.path.exists(calm_species_dir):
            import glob
            for fasta_file in glob.glob(os.path.join(calm_species_dir, "*.fasta")):
                with open(fasta_file) as f:
                    seq_lines = []
                    for line in f:
                        line = line.strip()
                        if line.startswith(">"):
                            if seq_lines:
                                seq = "".join(seq_lines).upper().replace(" ", "")
                                if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                                    sequences.append(seq)
                                seq_lines = []
                        else:
                            seq_lines.append(line)
                    if seq_lines:
                        seq = "".join(seq_lines).upper().replace(" ", "")
                        if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                            sequences.append(seq)
            print(f"After CaLM species data: {len(sequences)} CDS sequences")

        print(f"Total real CDS: {len(sequences)} sequences (no synthetic)")
        return sequences

    # data_source == "expanded" (default, backward compatible)
    expanded_file = os.path.expanduser(
        "./data/expanded_cds/expanded_cds_large.json"
    )
    if os.path.exists(expanded_file):
        with open(expanded_file) as f:
            data = json.load(f)
        sequences = data["sequences"]
        print(f"Loaded {len(sequences)} CDS sequences from expanded corpus")
        return sequences

    sequences = []

    clinvar_dir = os.path.expanduser("./data/task2_clinvar")
    cds_cache = os.path.join(clinvar_dir, "cds_sequences.json")
    if os.path.exists(cds_cache):
        with open(cds_cache) as f:
            data = json.load(f)
        for tid, seq in data.items():
            if isinstance(seq, str) and len(seq) >= 30:
                sequences.append(seq)
        print(f"After ClinVar cache: {len(sequences)} CDS sequences")

    calm_species_dir = "<MODEL_PATH>/cLMs/CaLM/data/species"
    if os.path.exists(calm_species_dir):
        import glob
        for fasta_file in glob.glob(os.path.join(calm_species_dir, "*.fasta")):
            with open(fasta_file) as f:
                seq_lines = []
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq_lines:
                            seq = "".join(seq_lines).upper().replace(" ", "")
                            if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                                sequences.append(seq)
                            seq_lines = []
                    else:
                        seq_lines.append(line)
                if seq_lines:
                    seq = "".join(seq_lines).upper().replace(" ", "")
                    if len(seq) >= 30 and all(c in "ATGC" for c in seq):
                        sequences.append(seq)
        print(f"After CaLM species data: {len(sequences)} CDS sequences")

    if len(sequences) < 1000:
        print(f"Only {len(sequences)} real CDS sequences. Supplementing with synthetic CDS.")
        np.random.seed(42)
        for _ in range(50000 - len(sequences)):
            length = np.random.randint(50, 400)
            seq_codons = [np.random.choice(CODON_VOCAB) for _ in range(length)]
            sequences.append("".join(seq_codons))
        print(f"After synthetic supplement: {len(sequences)} sequences")

    return sequences


def prepare_codon_sequences(sequences):
    """Convert raw CDS strings to space-separated codon format for codon tokenizer."""
    result = []
    for seq in sequences:
        codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
        valid = [c for c in codons if c in CODON_VOCAB]
        if len(valid) >= 10:
            result.append(" ".join(valid))
    return result


def prepare_char_sequences(sequences):
    """Keep raw CDS strings for character-level tokenizer."""
    result = []
    for seq in sequences:
        seq = seq.upper().replace(" ", "")
        if len(seq) >= 30:
            result.append(seq)
    return result


# ============================================================
# Pretraining
# ============================================================

def pretrain(tokenizer_type, num_epochs=30, effective_batch_size=64, output_suffix="", data_source="expanded"):
    """Pretrain a BERT model from scratch."""

    print(f"\n{'='*60}")
    print(f"Pretraining {tokenizer_type}-bert-ablation (data_source={data_source})")
    print(f"{'='*60}")

    sequences = load_cds_sequences(data_source)

    if tokenizer_type == "codon":
        tokenizer = CodonTokenizer()
        processed = prepare_codon_sequences(sequences)
        max_length = 512
        hidden_size = 512
        num_attention_heads = 8
        num_hidden_layers = 6
        intermediate_size = 2048
    else:
        tokenizer = CharTokenizer()
        processed = prepare_char_sequences(sequences)
        max_length = 1536
        hidden_size = 512
        num_attention_heads = 8
        num_hidden_layers = 6
        intermediate_size = 2048

    print(f"Processed {len(processed)} sequences")
    print(f"Max length: {max_length}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_length + 2,
        type_vocab_size=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )


    model = BertForMaskedLM(config)

    actual_params = sum(p.numel() for p in model.parameters())
    print(f"Actual params: {actual_params / 1e6:.1f}M")

    dataset = CDSDataset(processed, tokenizer, max_length)
    data_collator = DataCollatorForCDS(tokenizer, mlm_probability=0.15)

    output_dir = os.path.join(PRETRAIN_DIR, f"{tokenizer_type}-bert-ablation{output_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    gradient_accumulation = max(1, effective_batch_size // 8)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.06,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"Starting pretraining for {num_epochs} epochs...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Pretraining completed in {elapsed/3600:.1f} hours")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return output_dir


# ============================================================
# Evaluation
# ============================================================

def extract_embeddings_from_model(model, tokenizer, sequences, device, batch_size=8, layer=-2):
    """Extract embeddings from a pretrained model."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        encodings = tokenizer(
            batch_seqs,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[layer]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            embeddings = (hidden_states * mask_expanded).sum(1) / mask_expanded.sum(1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def run_lr_probing(X_train, y_train, X_test, y_test, C=1.0):
    """Logistic regression probing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc


def run_mlp_probing(X_train, y_train, X_test, y_test):
    """MLP probing."""
    from sklearn.metrics import roc_auc_score
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    auc = roc_auc_score(y_test, y_pred)
    return auc


def _dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def _parse_hgvs(name):
    import re
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)


CODON_CONTEXT = 15


def _build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds):
        return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = _dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None


def _balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        if n < 1:
            return df
        return pd.concat([
            df[df["label"] == 1].sample(n, random_state=42),
            df[df["label"] == 0].sample(n, random_state=42)
        ]).sample(frac=1, random_state=42)
    return df


def load_clinvar_data(task_name):
    """Load ClinVar classification data using same logic as run_mlp_independent_v2.py."""


    cds_file = "./data/task2_clinvar/cds_sequences.json"
    with open(cds_file) as f:
        cds_cache = json.load(f)

    clinvar_file = "./data/task2_clinvar/clinvar_raw.txt.gz"
    raw = pd.read_csv(clinvar_file, sep="\t", low_memory=False)

    snv = raw[raw["Type"] == "single nucleotide variant"].copy()

    pk, bk = ["Pathogenic", "Likely pathogenic"], ["Benign", "Likely benign"]

    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in pk):
            return 1
        if any(k in cs for k in bk):
            return 0
        return -1

    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]

    p = v["Name"].apply(_parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds_cache.keys()))
    var = v[h].copy()

    is_synonymous = "synonymous" in task_name
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))

    if is_synonymous:
        task_df = var[var["is_syn"]].copy()
        max_n = 2840
    else:
        task_df = var[~var["is_syn"]].copy()
        max_n = 5000

    task_df = _balance(task_df, max_n)

    sequences = []
    labels = []
    for _, row in task_df.iterrows():
        s = _build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s:
            sequences.append(s)
            labels.append(int(row["label"]))

    print(f"  Loaded {len(sequences)} sequences ({sum(labels)} patho, {len(labels)-sum(labels)} benign)")
    return sequences, np.array(labels)


def evaluate(tokenizer_type, output_suffix=""):
    """Evaluate a pretrained ablation model on Tasks 1 and 2."""

    print(f"\n{'='*60}")
    print(f"Evaluating {tokenizer_type}-bert-ablation")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(PRETRAIN_DIR, f"{tokenizer_type}-bert-ablation{output_suffix}")
    if not os.path.exists(model_dir):
        print(f"Model not found at {model_dir}. Run pretrain first.")
        return

    if tokenizer_type == "codon":
        tokenizer = CodonTokenizer()
    else:
        tokenizer = CharTokenizer()

    tokenizer.padding_side = "right"

    model = BertForMaskedLM.from_pretrained(model_dir)
    model.to(device)

    results = {}

    for task_name in ["task2_missense", "task3_synonymous"]:
        print(f"\n--- Evaluating on {task_name} ---")

        try:
            sequences, labels = load_clinvar_data(task_name)
        except Exception as e:
            print(f"  Failed to load data: {e}")
            continue

        if len(sequences) < 100:
            print(f"  Too few sequences ({len(sequences)}), skipping")
            continue

        if tokenizer_type == "codon":
            processed_seqs = sequences
        else:
            processed_seqs = [s.replace(" ", "") for s in sequences]

        embeddings = extract_embeddings_from_model(
            model, tokenizer, processed_seqs, device
        )

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_aucs_lr = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            auc = run_lr_probing(X_train, y_train, X_test, y_test)
            fold_aucs_lr.append(auc)

        mean_auc_lr = np.mean(fold_aucs_lr)
        std_auc_lr = np.std(fold_aucs_lr)

        print(f"LR AUC: {mean_auc_lr:.3f} +/- {std_auc_lr:.3f}")

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        mlp_auc = run_mlp_probing(X_train, y_train, X_test, y_test)
        print(f"MLP AUC: {mlp_auc:.3f}")

        results[task_name] = {
            "lr_auc_mean": float(mean_auc_lr),
            "lr_auc_std": float(std_auc_lr),
            "lr_auc_folds": [float(a) for a in fold_aucs_lr],
            "mlp_auc_test": float(mlp_auc),
            "tokenizer_type": tokenizer_type,
            "model_type": "from_scratch_bert",
        }

    result_file = os.path.join(RESULT_DIR, f"from_scratch_{tokenizer_type}{output_suffix}_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="From-scratch BERT ablation")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["pretrain", "eval", "all"],
                        help="pretrain only, eval only, or both")
    parser.add_argument("--tokenizer_type", type=str, default="both",
                        choices=["codon", "char", "both"],
                        help="which tokenizer type to train/eval")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="number of pretraining epochs")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="suffix for output dir (e.g., '-v2')")
    parser.add_argument("--data_source", type=str, default="expanded",
                        choices=["expanded", "real_only", "ensembl"],
                        help="CDS data source: expanded (55K w/ synthetic), real_only (4842), ensembl (downloaded)")
    args = parser.parse_args()

    tokenizer_types = ["codon", "char"] if args.tokenizer_type == "both" else [args.tokenizer_type]

    for tt in tokenizer_types:
        if args.mode in ["pretrain", "all"]:
            pretrain(tt, num_epochs=args.num_epochs, output_suffix=args.output_suffix, data_source=args.data_source)

        if args.mode in ["eval", "all"]:
            evaluate(tt, output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()