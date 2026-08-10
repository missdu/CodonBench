"""
Data-scale ablation: train codon-tokenized BERT (v3b architecture, 20M params)
on varying corpus sizes from the Ensembl CDS pool.

Fixed: architecture (6-layer, 512d, 20M), tokenizer (codon), epochs (30), seed (42)
Variable: corpus size (5K, 10K, 25K, 50K, 114K)

After training, extracts CLS embeddings and runs:
  1. Standard 5-fold CV (LR + MLP) on SynPath
  2. LOGO-CV (LR + MLP) on SynPath

Usage (on server):
  conda activate <CONDA_ENV>
  python scripts/run_data_scale_ablation.py --corpus_size 5000
  python scripts/run_data_scale_ablation.py --corpus_size 10000
  python scripts/run_data_scale_ablation.py --corpus_size 25000
  python scripts/run_data_scale_ablation.py --corpus_size 50000
  # 114K already exists as v3b
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["KERAS_BACKEND"] = "none"

import argparse
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

EXP_DIR = Path("./")
PRETRAIN_DIR = EXP_DIR / "from_scratch_models"
DATA_DIR = EXP_DIR / "data"
RESULT_DIR = EXP_DIR / "results" / "supplementary"

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


class CodonTokenizer(PreTrainedTokenizer):
    vocab_size = len(SPECIAL_TOKENS) + len(CODON_VOCAB)

    def __init__(self, **kwargs):
        self.vocab_dict = VOCAB
        self.id_to_token = {i: t for t, i in VOCAB.items()}
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
    def mask_token_id(self):
        return self.vocab_dict["[MASK]"]

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


class CDSDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer(
            seq, truncation=True, max_length=self.max_length,
            padding=False, return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": [1] * len(encoding["input_ids"]),
        }


class DataCollatorForCDS:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        max_len = max(len(e["input_ids"]) for e in examples)
        input_ids = []
        attention_mask = []
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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_ensembl_cds():
    ensembl_file = DATA_DIR / "ensembl_cds" / "ensembl_human_cds.json"
    with open(ensembl_file) as f:
        data = json.load(f)
    return data["sequences"]


def prepare_codon_sequences(sequences):
    result = []
    for seq in sequences:
        seq = seq.upper().replace(" ", "")
        if len(seq) < 30 or len(seq) % 3 != 0:
            continue
        valid = []
        for i in range(0, len(seq), 3):
            c = seq[i:i+3]
            if c in CODON_TABLE and CODON_TABLE[c] != '*':
                valid.append(c)
            else:
                valid = []
                break
        if len(valid) >= 10:
            result.append(" ".join(valid))
    return result


def subsample_sequences(sequences, n, seed=42):
    rng = np.random.RandomState(seed)
    if n >= len(sequences):
        return sequences
    indices = rng.choice(len(sequences), size=n, replace=False)
    return [sequences[i] for i in sorted(indices)]


def pretrain_model(corpus_size, num_epochs=30):
    suffix = f"-scale-cds{corpus_size}"
    output_dir = PRETRAIN_DIR / f"codon-bert-ablation{suffix}"

    if output_dir.exists() and (output_dir / "config.json").exists():
        print(f"Model already exists at {output_dir}, skipping pretraining.")
        return output_dir

    print(f"\n{'='*60}")
    print(f"Pretraining codon-bert-ablation{suffix}")
    print(f"Corpus size: {corpus_size} CDS, {num_epochs} epochs")
    print(f"{'='*60}")

    all_sequences = load_ensembl_cds()
    processed = prepare_codon_sequences(all_sequences)
    print(f"Total processed CDS: {len(processed)}")

    selected = subsample_sequences(processed, corpus_size)
    print(f"Selected: {len(selected)} CDS")

    tokenizer = CodonTokenizer()
    max_length = 512
    hidden_size = 512
    num_attention_heads = 8
    num_hidden_layers = 6
    intermediate_size = 2048

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
    print(f"Params: {actual_params / 1e6:.1f}M")

    dataset = CDSDataset(selected, tokenizer, max_length)
    data_collator = DataCollatorForCDS(tokenizer, mlm_probability=0.15)

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
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

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Pretraining completed in {elapsed/3600:.1f} hours")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

    return output_dir


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)


CODON_CONTEXT = 128


def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds):
        return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None


def load_clinvar_synpath():
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
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds_cache.keys()))
    var = v[h].copy()
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    task_df = var[var["is_syn"]].copy()
    max_n = 2840
    if len(task_df) > max_n:
        n = min(max_n // 2, int((task_df["label"] == 1).sum()), int((task_df["label"] == 0).sum()))
        task_df = pd.concat([
            task_df[task_df["label"] == 1].sample(n, random_state=42),
            task_df[task_df["label"] == 0].sample(n, random_state=42)
        ]).sample(frac=1, random_state=42)

    sequences = []
    labels = []
    tx_ids = []
    for _, row in task_df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s:
            sequences.append(s)
            labels.append(int(row["label"]))
            tx_ids.append(row["tx_id"])

    return sequences, np.array(labels), tx_ids, cds_cache


def extract_cls_embeddings(model_path, tokenizer, seqs, device="cuda:0", batch_size=16):
    model = BertForMaskedLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            all_ids = []
            for s in batch:
                enc = tokenizer(s, max_length=tokenizer.model_max_length)
                all_ids.append(enc["input_ids"])
            max_l = max(len(ids) for ids in all_ids)
            padded = [ids + [VOCAB["[PAD]"]] * (max_l - len(ids)) for ids in all_ids]
            input_ids = torch.tensor(padded, dtype=torch.long).to(device)
            attention_mask = (input_ids != VOCAB["[PAD]"]).long().to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)

    emb = np.concatenate(all_embs, axis=0)
    del model
    torch.cuda.empty_cache()
    return emb


def run_standard_cv(embeddings, labels, n_folds=5, seed=42):
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    lr_aucs = []
    mlp_aucs = []
    for train_idx, test_idx in skf.split(embeddings, labels):
        X_tr, X_te = embeddings[train_idx], embeddings[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(X_tr, y_tr)
        lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        lr_aucs.append(lr_auc)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            random_state=seed, early_stopping=True,
                            validation_fraction=0.1)
        mlp.fit(X_tr, y_tr)
        mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
        mlp_aucs.append(mlp_auc)

    return np.mean(lr_aucs), np.mean(mlp_aucs), np.mean(mlp_aucs) - np.mean(lr_aucs)


def run_logo_cv(embeddings, labels, tx_ids):
    unique_genes = list(set(tx_ids))
    gene_to_idx = defaultdict(list)
    for i, g in enumerate(tx_ids):
        gene_to_idx[g].append(i)

    valid_genes = [g for g in unique_genes
                   if len(set(labels[gene_to_idx[g]])) >= 2
                   and len(gene_to_idx[g]) >= 2]

    print(f"LOGO-CV: {len(valid_genes)} valid genes out of {len(unique_genes)}")

    lr_aucs = []
    mlp_aucs = []
    for gi, gene in enumerate(valid_genes):
        test_idx = gene_to_idx[gene]
        train_idx = [i for i in range(len(labels)) if i not in test_idx]

        if len(train_idx) < 10:
            continue

        X_tr, X_te = embeddings[train_idx], embeddings[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        if len(set(y_te)) < 2:
            continue

        lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        lr.fit(X_tr, y_tr)
        try:
            lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        except ValueError:
            continue

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                            random_state=42, early_stopping=True,
                            validation_fraction=0.1)
        mlp.fit(X_tr, y_tr)
        try:
            mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
        except ValueError:
            continue

        lr_aucs.append(lr_auc)
        mlp_aucs.append(mlp_auc)

        if (gi + 1) % 50 == 0:
            print(f"  LOGO fold {gi+1}/{len(valid_genes)}: "
                  f"LR={np.mean(lr_aucs):.3f}, MLP={np.mean(mlp_aucs):.3f}, "
                  f"gain={np.mean(mlp_aucs)-np.mean(lr_aucs):+.1f}pp")

    logo_lr = np.mean(lr_aucs) if lr_aucs else float('nan')
    logo_mlp = np.mean(mlp_aucs) if mlp_aucs else float('nan')
    logo_gain = (logo_mlp - logo_lr) * 100 if lr_aucs and mlp_aucs else float('nan')

    return logo_lr, logo_mlp, logo_gain, len(lr_aucs)


def main():
    parser = argparse.ArgumentParser(description="Data-scale ablation")
    parser.add_argument("--corpus_size", type=int, required=True,
                        help="Number of CDS sequences to use (5000, 10000, 25000, 50000)")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining if model already exists")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_path = pretrain_model(args.corpus_size, args.num_epochs)
    print(f"\nModel path: {model_path}")

    print("\nLoading SynPath data...")
    sequences, labels, tx_ids, cds_cache = load_clinvar_synpath()
    print(f"Loaded {len(sequences)} SynPath sequences, {len(set(tx_ids))} genes")

    print("\nExtracting CLS embeddings...")
    tokenizer = CodonTokenizer(max_length=512)
    embeddings = extract_cls_embeddings(model_path, tokenizer, sequences, device=device)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n=== Standard 5-fold CV ===")
    std_lr, std_mlp, std_gain = run_standard_cv(embeddings, labels)
    std_gain_pp = std_gain * 100
    print(f"LR AUC: {std_lr:.4f}, MLP AUC: {std_mlp:.4f}, Gain: {std_gain_pp:+.1f} pp")

    print("\n=== LOGO-CV ===")
    logo_lr, logo_mlp, logo_gain_pp, n_folds = run_logo_cv(embeddings, labels, tx_ids)
    print(f"LOGO-LR: {logo_lr:.4f}, LOGO-MLP: {logo_mlp:.4f}, Gain: {logo_gain_pp:+.1f} pp ({n_folds} folds)")

    results = {
        "corpus_size": args.corpus_size,
        "num_epochs": args.num_epochs,
        "architecture": "6-layer, 512d, 20M params",
        "standard_cv": {
            "lr_auc": float(std_lr),
            "mlp_auc": float(std_mlp),
            "gain_pp": float(std_gain_pp),
        },
        "logo_cv": {
            "lr_auc": float(logo_lr),
            "mlp_auc": float(logo_mlp),
            "gain_pp": float(logo_gain_pp),
            "n_valid_folds": int(n_folds),
        },
    }

    result_file = RESULT_DIR / f"data_scale_cds{args.corpus_size}_results.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    emb_file = RESULT_DIR / f"data_scale_cds{args.corpus_size}_emb.npy"
    np.save(emb_file, embeddings)
    print(f"Embeddings saved to {emb_file}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: corpus={args.corpus_size} CDS")
    print(f"  Standard CV: LR={std_lr:.3f}, MLP={std_mlp:.3f}, gain={std_gain_pp:+.1f} pp")
    print(f"  LOGO-CV:     LR={logo_lr:.3f}, MLP={logo_mlp:.3f}, gain={logo_gain_pp:+.1f} pp")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()