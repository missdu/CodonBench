"""CKA for CaLM and CodonTransformer using their specific APIs"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import src.models.xformers_compat
from src.eval.evaluation_utils import _fix_token_type_ids

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/cka_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


def linear_CKA(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic_xy = (XXT * YYT).sum()
    hsic_xx = (XXT * XXT).sum()
    hsic_yy = (YYT * YYT).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def rbf_CKA(X, Y, sigma=None):
    from scipy.spatial.distance import cdist
    if sigma is None:
        sigma = np.median(cdist(X, X))
    Kx = np.exp(-cdist(X, X)**2 / (2 * sigma**2))
    if sigma is None:
        sigma = np.median(cdist(Y, Y))
    Ky = np.exp(-cdist(Y, Y)**2 / (2 * sigma**2))
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()
    if hsic_xx == 0 or hsic_yy == 0:
        return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def extract_calm_embeddings(sequences, device="cuda:0"):
    from calm import CaLM
    calm = CaLM()
    all_embs = []
    batch_size = 8
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        rna_batch = [s.replace('T', 'U').replace('t', 'u') for s in batch]
        for seq in rna_batch:
            emb = calm.embed_sequence(seq)
            all_embs.append(emb.detach().cpu().numpy().flatten())
    return np.array(all_embs)


def extract_codontransformer_embeddings(sequences, device="cuda:0"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from Bio.Seq import Seq

    hf_id = "adibvafa/CodonTransformer"
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True).to(device).eval()

    aa_map = {'F':'F','L':'L','I':'I','M':'M','V':'V','S':'S','P':'P','T':'T',
              'A':'A','Y':'Y','H':'H','Q':'Q','N':'N','K':'K','D':'D','E':'E',
              'C':'C','W':'W','R':'R','G':'G'}

    all_embs = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            processed = []
            for seq in batch:
                codons = seq.split()
                aa_codons = []
                for c in codons:
                    if len(c) == 3 and c not in ['[PAD]', '[CLS]', '[SEP]']:
                        try:
                            aa = str(Seq(c).translate())
                        except:
                            aa = 'X'
                        aa_codons.append(f"{aa.lower()}_{c.lower()}")
                    else:
                        aa_codons.append(c)
                processed.append(" ".join(aa_codons))

            tokens = tokenizer(processed, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens = _fix_token_type_ids(tokens)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            out = model(**tokens, output_hidden_states=True)
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                emb = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            else:
                emb = out.logits

            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * attn_mask).sum(1) / attn_mask.sum(1)
            all_embs.append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embs)


def extract_codonbert_embeddings(sequences, device="cuda:0"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    local_path = "<MODEL_PATH>/cLMs/CodonBERT/codonbert"
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(local_path, trust_remote_code=True).to(device).eval()

    all_embs = []
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens = _fix_token_type_ids(tokens)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens, output_hidden_states=True)
            emb = out.hidden_states[-1]
            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * attn_mask).sum(1) / attn_mask.sum(1)
            all_embs.append(pooled.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_embs)


def main():
    device = "cuda:0"

    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()

    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw): return 1
        if any(k in cs for k in benign_kw): return 0
        return -1

    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    labeled = snv[snv["label"] >= 0].copy()
    valid = labeled[
        (labeled["ReferenceAlleleVCF"] != "na") &
        (labeled["AlternateAlleleVCF"] != "na") &
        (labeled["ReferenceAlleleVCF"].str.len() == 1) &
        (labeled["AlternateAlleleVCF"].str.len() == 1)
    ].copy()

    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])

    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()

    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))

    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_patho = int((task3_variants["label"]==1).sum())
    n_benign = int((task3_variants["label"]==0).sum())
    n_max = min(n_patho * 2, n_patho + n_benign)
    n_per_class = n_max // 2
    patho = task3_variants[task3_variants["label"]==1].sample(n=n_per_class, random_state=42)
    benign = task3_variants[task3_variants["label"]==0].sample(n=n_per_class, random_state=42)
    df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

    sequences = []
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            continue
        codon_seq = build_codon_context(cds_seq, cpos)
        if codon_seq is None:
            continue
        sequences.append(codon_seq)

    n_cka = min(2000, len(sequences))
    cka_sequences = sequences[:n_cka]
    print(f"  {n_cka} sequences for CKA")

    embeddings = {}

    print("\nExtracting CodonBERT embeddings...")
    try:
        embeddings["codonbert"] = extract_codonbert_embeddings(cka_sequences, device=device)
        print(f"  codonbert: {embeddings['codonbert'].shape}")
    except Exception as e:
        print(f"  codonbert FAILED: {e}")

    print("\nExtracting CodonTransformer embeddings...")
    try:
        embeddings["codontransformer"] = extract_codontransformer_embeddings(cka_sequences, device=device)
        print(f"  codontransformer: {embeddings['codontransformer'].shape}")
    except Exception as e:
        print(f"  codontransformer FAILED: {e}")

    print("\nExtracting CaLM embeddings...")
    try:
        embeddings["calm"] = extract_calm_embeddings(cka_sequences, device=device)
        print(f"  calm: {embeddings['calm'].shape}")
    except Exception as e:
        print(f"  calm FAILED: {e}")

    pairs = [
        ("codontransformer", "codonbert"),
        ("calm", "codonbert"),
    ]

    print(f"\n{'='*60}")
    print("CKA Analysis")
    print(f"{'='*60}")

    existing = []
    try:
        with open(OUT_DIR / "cka_extended_results.json") as f:
            existing = json.load(f)
    except:
        pass

    results = list(existing)
    for m1, m2 in pairs:
        if m1 not in embeddings or m2 not in embeddings:
            print(f"  {m1} vs {m2}: SKIPPED")
            continue
        X = embeddings[m1]
        Y = embeddings[m2]
        if X.shape[0] != Y.shape[0]:
            n = min(X.shape[0], Y.shape[0])
            X = X[:n]
            Y = Y[:n]

        lin_cka = linear_CKA(X, Y)
        rbf_cka = rbf_CKA(X, Y)

        r = {"model1": m1, "model2": m2, "task": "task3_synonymous",
             "n_samples": X.shape[0], "linear_cka": round(lin_cka, 4),
             "rbf_cka": round(rbf_cka, 4)}
        results.append(r)
        print(f"  {m1} vs {m2}: Linear={lin_cka:.4f}, RBF={rbf_cka:.4f}")

    with open(OUT_DIR / "cka_extended_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved ({len(results)} pairs)")


if __name__ == "__main__":
    main()