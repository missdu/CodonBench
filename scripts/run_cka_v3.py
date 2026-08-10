"""CKA extended analysis - fixed version.
Fixes: 1) CaLM pip package loading, 2) EnCodon AutoModel fallback, 3) GPU memory management, 4) Proxy unset
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "<MODEL_PATH>/cLMs/CaLM")

import os
for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import re
import gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import src.models.xformers_compat
from src.eval.evaluation_utils import _fix_token_type_ids

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/cka_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE = OUT_DIR / "embeddings"
EMB_CACHE.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16

MODELS = {
    "codonbert": {
        "local_path": "<MODEL_PATH>/cLMs/CodonBERT/codonbert",
        "use_mlm": True, "rna": False, "token_type": "codon_dna",
    },
    "codonbert_hf": {
        "hf_id": "lhallee/CodonBERT",
        "use_mlm": True, "rna": True, "token_type": "codon_rna",
    },
    "encodon-80m": {
        "hf_id": "goodarzilab/encodon-80M",
        "use_mlm": True, "rna": False, "token_type": "codon_dna",
    },
    "encodon-620m": {
        "hf_id": "goodarzilab/encodon-620M",
        "use_mlm": True, "rna": False, "token_type": "codon_dna",
    },
    "calm": {
        "is_calm": True, "use_pip_package": True,
        "use_mlm": False, "rna": True, "token_type": "codon_rna",
    },
    "codontransformer": {
        "hf_id": "adibvafa/CodonTransformer",
        "use_mlm": True, "rna": False, "token_type": "aa_codon",
        "is_codontransformer": True,
    },
}

CKA_PAIRS = [
    ("encodon-620m", "encodon-80m"),
    ("encodon-620m", "codonbert"),
    ("encodon-620m", "codonbert_hf"),
    ("encodon-620m", "codontransformer"),
    ("encodon-620m", "calm"),
    ("codontransformer", "codonbert"),
    ("codontransformer", "codonbert_hf"),
    ("codontransformer", "encodon-80m"),
    ("codontransformer", "calm"),
    ("calm", "codonbert"),
    ("calm", "codonbert_hf"),
    ("calm", "encodon-80m"),
    ("codonbert", "codonbert_hf"),
    ("codonbert", "encodon-80m"),
    ("codonbert_hf", "encodon-80m"),
]


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


def extract_embeddings(model_name, model_info, codon_seqs, device="cuda:1"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    from Bio.Seq import Seq

    is_calm = model_info.get("is_calm", False)
    is_ct = model_info.get("is_codontransformer", False)
    rna = model_info.get("rna", False)
    use_pip = model_info.get("use_pip_package", False)

    hf_id = model_info.get("hf_id")
    local_path = model_info.get("local_path")
    model_path = local_path if local_path else hf_id

    print(f"  Loading {model_name} from {model_path}...")
    tokenizer = None
    model = None

    if is_calm and use_pip:
        from calm import CaLM
        calm_model = CaLM()
        model = calm_model.model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded: {type(model).__name__}, {params_m:.1f}M params")

    all_embs = []
    batch_size = 4

    with torch.no_grad():
        for i in range(0, len(codon_seqs), batch_size):
            batch = codon_seqs[i:i+batch_size]

            if is_calm:
                rna_batch = [seq.replace('T', 'U').replace('t', 'u') for seq in batch]
                embs_calm = calm_model.embed_sequences(rna_batch)
                all_embs.append(embs_calm.cpu().numpy())
                continue
            elif is_ct:
                aa_map = {'F':'F','L':'L','I':'I','M':'M','V':'V','S':'S','P':'P','T':'T','A':'A','Y':'Y','H':'H','Q':'Q','N':'N','K':'K','D':'D','E':'E','C':'C','W':'W','R':'R','G':'G'}
                processed = []
                for seq in batch:
                    codons = seq.split()
                    aa_codons = []
                    for c in codons:
                        if len(c) == 3 and c not in ('[PAD]', '[CLS]', '[SEP]'):
                            aa = aa_map.get(str(Seq(c).translate()), 'X')
                            aa_codons.append(f"{aa.lower()}_{c.lower()}")
                        else:
                            aa_codons.append(c)
                    processed.append(" ".join(aa_codons))
                batch = processed
            elif rna:
                batch = [seq.replace('T', 'U').replace('t', 'u') for seq in batch]

            if not is_calm:
                tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                tokens = _fix_token_type_ids(tokens)
                tokens = {k: v.to(device) for k, v in tokens.items()}

            try:
                out = model(**tokens, output_hidden_states=True)
            except Exception:
                out = model(**tokens)

            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                emb = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            else:
                emb = out.logits

            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * attn_mask).sum(1) / attn_mask.sum(1)
            all_embs.append(pooled.cpu().numpy())

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return np.vstack(all_embs)


def main():
    device = "cuda:1"

    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")

    print("Loading ClinVar...")
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
    labels = []
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
        labels.append(row["label"])

    labels = np.array(labels, dtype=int)
    print(f"  {len(sequences)} SynPath samples, P={int(labels.sum())} B={int(len(labels)-labels.sum())}")

    n_cka = min(2000, len(sequences))
    cka_sequences = sequences[:n_cka]
    cka_labels = labels[:n_cka]

    # Save sequences for reproducibility
    with open(EMB_CACHE / "cka_sequences.json", "w") as f:
        json.dump({"sequences": cka_sequences, "labels": cka_labels.tolist(), "n": n_cka}, f)

    # Extract embeddings one model at a time, cache to disk
    embeddings = {}
    for mname, minfo in MODELS.items():
        emb_file = EMB_CACHE / f"{mname}_emb.npy"
        if emb_file.exists():
            print(f"\n--- {mname}: loading cached embeddings ---")
            embeddings[mname] = np.load(str(emb_file))
            print(f"  Shape: {embeddings[mname].shape}")
            continue

        print(f"\n--- {mname}: extracting embeddings ---")
        try:
            emb = extract_embeddings(mname, minfo, cka_sequences, device=device)
            embeddings[mname] = emb
            np.save(str(emb_file), emb)
            print(f"  Shape: {emb.shape}, saved to {emb_file}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")

    print(f"\n{'='*60}")
    print("CKA Analysis")
    print(f"{'='*60}")

    results = []
    for m1, m2 in CKA_PAIRS:
        if m1 not in embeddings or m2 not in embeddings:
            print(f"  {m1} vs {m2}: SKIPPED (missing embeddings)")
            continue

        X = embeddings[m1]
        Y = embeddings[m2]

        lin_cka = linear_CKA(X, Y)
        rbf_cka = rbf_CKA(X, Y)

        r = {"model1": m1, "model2": m2, "task": "task3_synonymous",
             "n_samples": n_cka, "linear_cka": round(lin_cka, 4),
             "rbf_cka": round(rbf_cka, 4)}
        results.append(r)
        print(f"  {m1} vs {m2}: Linear={lin_cka:.4f}, RBF={rbf_cka:.4f}")

    out_file = OUT_DIR / "cka_extended_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()