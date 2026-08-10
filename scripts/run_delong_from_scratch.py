"""
DeLong paired AUC test for from-scratch ablation models.

For each version (v1, v3a, v3b), loads codon and char models,
extracts embeddings on the SAME data, runs LR and MLP probing
on the SAME train/test split, and computes DeLong 95% CI for
the AUC difference (codon - char).

Usage:
  python run_delong_from_scratch.py --versions v1 v3a v3b --gpu 0
  python run_delong_from_scratch.py --versions v3b --gpu 1
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import BertForMaskedLM, PreTrainedTokenizer

RESULT_DIR = os.path.expanduser("./results")
PRETRAIN_DIR = os.path.expanduser("./from_scratch_models")

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
        kwargs.setdefault("mask_token", "[MASK]")
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
    def is_fast(self):
        return False


class CharTokenizer(PreTrainedTokenizer):
    vocab_size = len(SPECIAL_TOKENS) + 4

    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + ["A", "T", "G", "C"])}
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
        return list(text.upper().replace(" ", ""))

    def _convert_token_to_id(self, token):
        return self.vocab_dict.get(token, self.vocab_dict["[UNK]"])

    def _convert_id_to_token(self, index):
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self):
        return dict(self.vocab_dict)

    @property
    def is_fast(self):
        return False


CODON_CONTEXT = 15


def _dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def _parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)


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


def load_clinvar_synonymous():
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

    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    task_df = var[var["is_syn"]].copy()
    task_df = _balance(task_df, 2840)

    sequences = []
    labels = []
    for _, row in task_df.iterrows():
        s = _build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s:
            sequences.append(s)
            labels.append(int(row["label"]))

    print(f"Loaded {len(sequences)} synonymous variants ({sum(labels)} patho, {len(labels)-sum(labels)} benign)")
    return sequences, np.array(labels)


def extract_embeddings(model, tokenizer, sequences, device, batch_size=8, layer=-2):
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


def delong_roc_variance(y_true, y_scores):
    """Compute DeLong variance for AUC estimation."""
    y_true = np.asarray(y_true, dtype=bool)
    y_scores = np.asarray(y_scores, dtype=float)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0


    pos_scores = y_scores[y_true]
    neg_scores = y_scores[~y_true]

    V10 = np.zeros(n_pos)
    for i, sp in enumerate(pos_scores):
        V10[i] = np.mean((neg_scores < sp).astype(float) + 0.5 * (neg_scores == sp).astype(float))

    V01 = np.zeros(n_neg)
    for j, sn in enumerate(neg_scores):
        V01[j] = np.mean((pos_scores > sn).astype(float) + 0.5 * (pos_scores == sn).astype(float))

    S10 = np.var(V10, ddof=1)
    S01 = np.var(V01, ddof=1)

    var_auc = S10 / n_pos + S01 / n_neg
    return var_auc


def delong_paired_test(y_true, y_scores_a, y_scores_b):
    """
    DeLong paired test for two correlated AUCs.
    Returns: (auc_a, auc_b, delta, z_stat, p_value, ci_lower, ci_upper)
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_scores_a = np.asarray(y_scores_a, dtype=float)
    y_scores_b = np.asarray(y_scores_b, dtype=float)

    auc_a = roc_auc_score(y_true, y_scores_a)
    auc_b = roc_auc_score(y_true, y_scores_b)
    delta = auc_a - auc_b

    var_a = delong_roc_variance(y_true, y_scores_a)
    var_b = delong_roc_variance(y_true, y_scores_b)

    cov_ab = _delong_covariance(y_true, y_scores_a, y_scores_b)

    var_delta = var_a + var_b - 2 * cov_ab

    if var_delta <= 0:
        print(f"  Warning: negative variance of delta ({var_delta:.6f}), using absolute value")
        var_delta = abs(var_delta)

    se_delta = np.sqrt(var_delta)
    z_stat = delta / se_delta if se_delta > 0 else 0.0

    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    ci_lower = delta - 1.96 * se_delta
    ci_upper = delta + 1.96 * se_delta

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta": float(delta),
        "se_delta": float(se_delta),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "ci_lower_95": float(ci_lower),
        "ci_upper_95": float(ci_upper),
        "var_a": float(var_a),
        "var_b": float(var_b),
        "cov_ab": float(cov_ab),
    }


def _delong_covariance(y_true, y_scores_a, y_scores_b):
    """Compute covariance between two AUC estimates (DeLong method)."""
    y_true = np.asarray(y_true, dtype=bool)
    y_scores_a = np.asarray(y_scores_a, dtype=float)
    y_scores_b = np.asarray(y_scores_b, dtype=float)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    pos_scores_a = y_scores_a[y_true]
    neg_scores_a = y_scores_a[~y_true]
    pos_scores_b = y_scores_b[y_true]
    neg_scores_b = y_scores_b[~y_true]

    V10_a = np.zeros(n_pos)
    V10_b = np.zeros(n_pos)
    for i in range(n_pos):
        V10_a[i] = np.mean((neg_scores_a < pos_scores_a[i]).astype(float) +
                           0.5 * (neg_scores_a == pos_scores_a[i]).astype(float))
        V10_b[i] = np.mean((neg_scores_b < pos_scores_b[i]).astype(float) +
                           0.5 * (neg_scores_b == pos_scores_b[i]).astype(float))

    V01_a = np.zeros(n_neg)
    V01_b = np.zeros(n_neg)
    for j in range(n_neg):
        V01_a[j] = np.mean((pos_scores_a > neg_scores_a[j]).astype(float) +
                           0.5 * (pos_scores_a == neg_scores_a[j]).astype(float))
        V01_b[j] = np.mean((pos_scores_b > neg_scores_b[j]).astype(float) +
                           0.5 * (pos_scores_b == neg_scores_b[j]).astype(float))

    cov_V10 = np.cov(V10_a, V10_b, ddof=1)[0, 1]
    cov_V01 = np.cov(V01_a, V01_b, ddof=1)[0, 1]

    return cov_V10 / n_pos + cov_V01 / n_neg


def run_lr_and_get_probs(X_train, y_train, X_test, y_test, C=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    return y_pred


def run_mlp_and_get_probs(X_train, y_train, X_test, y_test, hidden_dim=128, epochs=100):
    from sklearn.metrics import roc_auc_score
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    input_dim = X_train.shape[1]
    mlp = nn.Sequential(
        nn.Linear(input_dim, 128),
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
    return y_pred


VERSION_CONFIG = {
    "v1": {"codon_suffix": "", "char_suffix": ""},
    "v2": {"codon_suffix": "-v2", "char_suffix": "-v2"},
    "v3a": {"codon_suffix": "-v3a-real100ep", "char_suffix": "-v3a-real100ep"},
    "v3b": {"codon_suffix": "-v3b-ensembl", "char_suffix": "-v3b-ensembl"},
    "v4": {"codon_suffix": "-v4-scale110m", "char_suffix": "-v4-scale110m"},
}


def main():
    parser = argparse.ArgumentParser(description="DeLong paired AUC test for from-scratch ablation")
    parser.add_argument("--versions", nargs="+", default=["v1", "v3a", "v3b"],
                        choices=["v1", "v2", "v3a", "v3b", "v4"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task", type=str, default="synonymous",
                        choices=["missense", "synonymous", "both"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sequences, labels = load_clinvar_synonymous()

    all_results = {}

    for version in args.versions:
        print(f"\n{'='*60}")
        print(f"Version: {version}")
        print(f"{'='*60}")

        cfg = VERSION_CONFIG[version]
        codon_dir = os.path.join(PRETRAIN_DIR, f"codon-bert-ablation{cfg['codon_suffix']}")
        char_dir = os.path.join(PRETRAIN_DIR, f"char-bert-ablation{cfg['char_suffix']}")

        if not os.path.exists(codon_dir):
            print(f"  Codon model not found: {codon_dir}, skipping")
            continue
        if not os.path.exists(char_dir):
            print(f"  Char model not found: {char_dir}, skipping")
            continue

        codon_tokenizer = CodonTokenizer()
        char_tokenizer = CharTokenizer()
        codon_tokenizer.padding_side = "right"
        char_tokenizer.padding_side = "right"

        print("  Loading codon model...")
        codon_model = BertForMaskedLM.from_pretrained(codon_dir).to(device)
        print("  Loading char model...")
        char_model = BertForMaskedLM.from_pretrained(char_dir).to(device)

        codon_seqs = sequences
        char_seqs = [s.replace(" ", "") for s in sequences]

        print("  Extracting codon embeddings...")
        emb_codon = extract_embeddings(codon_model, codon_tokenizer, codon_seqs, device)
        print("  Extracting char embeddings...")
        emb_char = extract_embeddings(char_model, char_tokenizer, char_seqs, device)

        del codon_model
        del char_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"  Codon embeddings: {emb_codon.shape}")
        print(f"  Char embeddings: {emb_char.shape}")

        from sklearn.model_selection import StratifiedKFold, train_test_split

        version_results = {}

        for probe_type in ["lr", "mlp"]:
            print(f"\n  --- {probe_type.upper()} Probing ---")

            if probe_type == "lr":
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_results = []

                for fold, (train_idx, test_idx) in enumerate(skf.split(emb_codon, labels)):
                    X_train_c, X_test_c = emb_codon[train_idx], emb_codon[test_idx]
                    X_train_h, X_test_h = emb_char[train_idx], emb_char[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]

                    probs_c = run_lr_and_get_probs(X_train_c, y_train, X_test_c, y_test)
                    probs_h = run_lr_and_get_probs(X_train_h, y_train, X_test_h, y_test)

                    result = delong_paired_test(y_test, probs_c, probs_h)
                    fold_results.append(result)

                    print(f"    Fold {fold+1}: codon AUC={result['auc_a']:.4f}, "
                          f"char AUC={result['auc_b']:.4f}, "
                          f"Δ={result['delta']:+.4f}, "
                          f"95% CI=[{result['ci_lower_95']:+.4f}, {result['ci_upper_95']:+.4f}], "
                          f"p={result['p_value']:.4e}")

                mean_delta = np.mean([r["delta"] for r in fold_results])
                mean_ci_lower = np.mean([r["ci_lower_95"] for r in fold_results])
                mean_ci_upper = np.mean([r["ci_upper_95"] for r in fold_results])
                mean_p = np.mean([r["p_value"] for r in fold_results])

                print(f"\n  LR Summary: mean Δ={mean_delta:+.4f}, "
                      f"mean 95% CI=[{mean_ci_lower:+.4f}, {mean_ci_upper:+.4f}], "
                      f"mean p={mean_p:.4e}")

                version_results["lr"] = {
                    "fold_results": fold_results,
                    "mean_delta": float(mean_delta),
                    "mean_ci_lower": float(mean_ci_lower),
                    "mean_ci_upper": float(mean_ci_upper),
                    "mean_p": float(mean_p),
                }

            else:
                indices = np.arange(len(labels))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=42, stratify=labels
                )

                X_train_c = emb_codon[train_idx]
                X_test_c = emb_codon[test_idx]
                X_train_h = emb_char[train_idx]
                X_test_h = emb_char[test_idx]
                y_train = labels[train_idx]
                y_test = labels[test_idx]

                probs_c = run_mlp_and_get_probs(X_train_c, y_train, X_test_c, y_test)
                probs_h = run_mlp_and_get_probs(X_train_h, y_train, X_test_h, y_test)

                result = delong_paired_test(y_test, probs_c, probs_h)

                print(f"  MLP: codon AUC={result['auc_a']:.4f}, "
                      f"char AUC={result['auc_b']:.4f}, "
                      f"Δ={result['delta']:+.4f}, "
                      f"95% CI=[{result['ci_lower_95']:+.4f}, {result['ci_upper_95']:+.4f}], "
                      f"p={result['p_value']:.4e}")

                version_results["mlp"] = result

        all_results[version] = version_results

    output_file = os.path.join(RESULT_DIR, "delong_from_scratch_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for version, vres in all_results.items():
        print(f"\n{version}:")
        if "lr" in vres:
            lr = vres["lr"]
            print(f"  LR:  Δ={lr['mean_delta']:+.4f}, 95% CI=[{lr['mean_ci_lower']:+.4f}, {lr['mean_ci_upper']:+.4f}], p={lr['mean_p']:.4e}")
        if "mlp" in vres:
            mlp = vres["mlp"]
            print(f"  MLP: Δ={mlp['delta']:+.4f}, 95% CI=[{mlp['ci_lower_95']:+.4f}, {mlp['ci_upper_95']:+.4f}], p={mlp['p_value']:.4e}")


if __name__ == "__main__":
    main()