"""
Exp3: From-scratch scale expansion 鈥?12-layer 768-dim BERT (~110M params).

Tests whether the tokenization advantage from the 20M-parameter ablation
scales to production-model size. Uses the same v3b Ensembl corpus (114K CDS).

Usage:
  python run_exp3_scale_expansion.py --mode pretrain --tokenizer_type codon
  python run_exp3_scale_expansion.py --mode pretrain --tokenizer_type char
  python run_exp3_scale_expansion.py --mode eval --tokenizer_type codon
  python run_exp3_scale_expansion.py --mode eval --tokenizer_type char
  python run_exp3_scale_expansion.py --mode all
"""

import argparse
import json
import os
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
ENSEMBL_DATA = os.path.expanduser("./data/ensembl_cds/ensembl_human_cds.json")

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
        kwargs.setdefault("mask_token", SPECIAL_TOKENS[4])
        super().__init__(**kwargs)

    @property
    def is_fast(self):
        return False

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab_dict.get(token, self.vocab_dict.get("[UNK]"))

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_vocab(self):
        return dict(self.vocab_dict)

    @property
    def vocab_size(self):
        return len(self.vocab_dict)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = os.path.join(save_directory, (filename_prefix or "") + "vocab.txt")
        with open(path, "w") as f:
            for token, idx in sorted(self.vocab_dict.items(), key=lambda x: x[1]):
                f.write(token + "\n")
        return (path,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [0] * len(token_ids_0)
        return [1] + [0] * len(token_ids_0) + [1]


CHAR_VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "T", "G", "C"]


class CharTokenizer(PreTrainedTokenizer):
    vocab_size = len(CHAR_VOCAB)

    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(CHAR_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 1536)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        super().__init__(**kwargs)

    @property
    def is_fast(self):
        return False

    def _tokenize(self, text):
        return list(text.replace(" ", ""))

    def _convert_token_to_id(self, token):
        return self.vocab_dict.get(token, self.vocab_dict.get("[UNK]"))

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self):
        return dict(self.vocab_dict)

    @property
    def vocab_size(self):
        return len(self.vocab_dict)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = os.path.join(save_directory, (filename_prefix or "") + "vocab.txt")
        with open(path, "w") as f:
            for token, idx in sorted(self.vocab_dict.items(), key=lambda x: x[1]):
                f.write(token + "\n")
        return (path,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [0] * len(token_ids_0)
        return [1] + [0] * len(token_ids_0) + [1]


def load_ensembl_cds():
    if not os.path.exists(ENSEMBL_DATA):
        print(f"ERROR: Ensembl data not found at {ENSEMBL_DATA}")
        sys.exit(1)
    with open(ENSEMBL_DATA, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        sequences = data.get("sequences", [])
    elif isinstance(data, list):
        sequences = [entry.get("sequence", entry) if isinstance(entry, dict) else entry for entry in data]
    else:
        sequences = []
    valid = []
    for seq in sequences:
        if isinstance(seq, str) and len(seq) >= 30 and len(seq) % 3 == 0:
            valid.append(seq.upper())
    print(f"Loaded {len(valid)} Ensembl CDS sequences (from {len(sequences)} total)")
    return valid


def prepare_codon_sequences(sequences):
    processed = []
    for seq in sequences:
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        valid = all(c in CODON_VOCAB for c in codons)
        if valid:
            processed.append(" ".join(codons))
    print(f"Codon-processed: {len(processed)} valid sequences (from {len(sequences)})")
    return processed


def prepare_char_sequences(sequences):
    processed = []
    for seq in sequences:
        valid = all(c in "ATGC" for c in seq)
        if valid:
            processed.append(seq)
    print(f"Char-processed: {len(processed)} valid sequences (from {len(sequences)})")
    return processed


class CDSDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        input_ids = [self.tokenizer.cls_token_id] + \
                    [self.tokenizer._convert_token_to_id(t) for t in tokens] + \
                    [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def tokenize(self, text):
        return self.tokenizer._tokenize(text)


class DataCollatorForCDS:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        batch = {
            "input_ids": torch.stack([e["input_ids"] for e in examples]),
            "attention_mask": torch.stack([e["attention_mask"] for e in examples]),
        }
        labels = batch["input_ids"].clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = self.tokenizer._convert_token_to_id("[MASK]")
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        batch["input_ids"][indices_random] = random_words[indices_random]
        batch["labels"] = labels
        return batch


def pretrain(tokenizer_type, num_epochs=30, output_suffix=""):
    print(f"\n{'='*60}")
    print(f"Exp3: Pretraining {tokenizer_type}-bert-ablation{output_suffix}")
    print(f"Architecture: 12 layers, 768 dim, 12 heads (~110M params)")
    print(f"{'='*60}")

    sequences = load_ensembl_cds()

    if tokenizer_type == "codon":
        tokenizer = CodonTokenizer()
        processed = prepare_codon_sequences(sequences)
        max_length = 512
        hidden_size = 768
        num_attention_heads = 12
        num_hidden_layers = 12
        intermediate_size = 3072
    else:
        tokenizer = CharTokenizer()
        processed = prepare_char_sequences(sequences)
        max_length = 1536
        hidden_size = 768
        num_attention_heads = 12
        num_hidden_layers = 12
        intermediate_size = 3072

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

    effective_batch_size = 32
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

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nPretraining completed in {elapsed/3600:.1f} hours")

    trainer.save_model(output_dir)
    tokenizer.save_vocabulary(output_dir)
    print(f"Model saved to {output_dir}")

    meta = {
        "tokenizer_type": tokenizer_type,
        "num_params_M": actual_params / 1e6,
        "num_sequences": len(processed),
        "num_epochs": num_epochs,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "max_length": max_length,
        "training_time_hours": elapsed / 3600,
        "data_source": "ensembl_114K",
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def evaluate(tokenizer_type, output_suffix=""):
    print(f"\n{'='*60}")
    print(f"Exp3: Evaluating {tokenizer_type}-bert-ablation{output_suffix}")
    print(f"{'='*60}")

    model_dir = os.path.join(PRETRAIN_DIR, f"{tokenizer_type}-bert-ablation{output_suffix}")
    if not os.path.exists(model_dir):
        print(f"ERROR: Model not found at {model_dir}")
        return

    if tokenizer_type == "codon":
        tokenizer = CodonTokenizer()
        max_length = 512
    else:
        tokenizer = CharTokenizer()
        max_length = 1536

    config = BertConfig.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded from {model_dir}, device={device}")

    clinvar_dir = os.path.expanduser("./data/task2_clinvar")
    cds_cache = os.path.join(clinvar_dir, "cds_sequences.json")

    with open(cds_cache, "r") as f:
        cds_data = json.load(f)

    task2_file = os.path.join(clinvar_dir, "clinvar_variants_task2_missense.json")
    task3_file = os.path.join(clinvar_dir, "clinvar_variants_task3_synonymous.json")

    results = {}

    for task_file, task_name in [(task2_file, "task2_missense"), (task3_file, "task3_synonymous")]:
        if not os.path.exists(task_file):
            print(f"WARNING: {task_file} not found, skipping")
            continue

        with open(task_file, "r") as f:
            variants = json.load(f)

        print(f"\nEvaluating {task_name}: {len(variants)} variants")

        embeddings_list = []
        labels_list = []

        for i, var in enumerate(variants):
            if i % 500 == 0:
                print(f"  Processing variant {i}/{len(variants)}")

            tx_id = var.get("transcript_id", var.get("tx_id", ""))
            pos = var.get("codon_position", var.get("position", 0))
            label = var.get("label", var.get("pathogenic", 0))

            cds_seq = cds_data.get(tx_id, "")
            if not cds_seq:
                continue

            codon_idx = (pos - 1) // 3
            start = max(0, codon_idx - 16)
            end = min(len(cds_seq) // 3, codon_idx + 17)
            subseq = cds_seq[start * 3: end * 3]

            if tokenizer_type == "codon":
                codons = [subseq[j:j+3] for j in range(0, len(subseq), 3)]
                text = " ".join(codons)
            else:
                text = subseq

            tokens = tokenizer._tokenize(text)
            if len(tokens) > max_length - 2:
                tokens = tokens[:max_length - 2]

            input_ids = [tokenizer.cls_token_id] + \
                        [tokenizer._convert_token_to_id(t) for t in tokens] + \
                        [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

            input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(device)
            attention_mask_t = torch.tensor([attention_mask], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids_t, attention_mask=attention_mask_t, output_hidden_states=True)
                cls_emb = outputs.hidden_states[-1][0, 0].cpu().numpy()

            embeddings_list.append(cls_emb)
            labels_list.append(label)

        if not embeddings_list:
            print(f"  No valid embeddings for {task_name}")
            continue

        X = np.array(embeddings_list)
        y = np.array(labels_list)
        print(f"  Extracted {len(X)} embeddings, positive rate: {y.mean():.3f}")

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            lr.fit(X_train, y_train)
            y_pred = lr.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            fold_aucs.append(auc)

        lr_mean = np.mean(fold_aucs)
        lr_std = np.std(fold_aucs)
        print(f"  LR 5-fold CV AUC: {lr_mean:.3f} 卤 {lr_std:.3f}")
        print(f"  Folds: {[f'{a:.3f}' for a in fold_aucs]}")

        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                           early_stopping=True, validation_fraction=0.1)
        mlp.fit(X_train, y_train)
        y_pred_mlp = mlp.predict_proba(X_test)[:, 1]
        mlp_auc = roc_auc_score(y_test, y_pred_mlp)
        print(f"  MLP test AUC: {mlp_auc:.3f}")

        results[task_name] = {
            "n_samples": len(X),
            "positive_rate": float(y.mean()),
            "lr_5fold_auc": float(lr_mean),
            "lr_5fold_std": float(lr_std),
            "lr_fold_aucs": [float(a) for a in fold_aucs],
            "mlp_test_auc": float(mlp_auc),
        }

    result_file = os.path.join(RESULT_DIR, f"from_scratch_{tokenizer_type}{output_suffix}_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Exp3: From-scratch scale expansion (12L 768D)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["pretrain", "eval", "all"])
    parser.add_argument("--tokenizer_type", type=str, default="both",
                        choices=["codon", "char", "both"])
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--output_suffix", type=str, default="-v4-scale110m")
    args = parser.parse_args()

    tokenizer_types = ["codon", "char"] if args.tokenizer_type == "both" else [args.tokenizer_type]

    for tt in tokenizer_types:
        if args.mode in ["pretrain", "all"]:
            pretrain(tt, num_epochs=args.num_epochs, output_suffix=args.output_suffix)
        if args.mode in ["eval", "all"]:
            evaluate(tt, output_suffix=args.output_suffix)


if __name__ == "__main__":
    main()