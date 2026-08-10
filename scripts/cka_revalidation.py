"""Quick CKA revalidation: recompute CKA for original 3 pairs using extended script's method."""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json, re, numpy as np, pandas as pd, torch
from pathlib import Path
import src.models.xformers_compat
from src.eval.evaluation_utils import _fix_token_type_ids

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    cpos = int(cpos)
    if cpos < 1 or cpos > len(cds_seq): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - context_codons), min(len(cds_seq) // 3, ci + context_codons + 1)
    codons = dna_to_codons(cds_seq[s*3:e*3])
    return " ".join(codons) if codons else None

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def linear_CKA(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    hsic_xy = (XXT * YYT).sum()
    hsic_xx = (XXT * XXT).sum()
    hsic_yy = (YYT * YYT).sum()
    if hsic_xx == 0 or hsic_yy == 0: return 0.0
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
    if hsic_xx == 0 or hsic_yy == 0: return 0.0
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))

def get_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=8, max_length=512, rna=False):
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        if rna:
            batch = [s.replace("T", "U").replace("t", "u") for s in batch]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = _fix_token_type_ids(inputs)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

def load_task3(cds_cache, raw_data, n=2000):
    snv = raw_data[raw_data["Type"] == "single nucleotide variant"].copy()
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
    df = variants[variants["is_syn"]].copy()
    n_pc = min(n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
    df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                    df[df["label"]==0].sample(n_pc, random_state=42)])
    seqs = []
    for _, row in df.iterrows():
        s = build_codon_context(cds_cache[row["tx_id"]], row["cpos"])
        if s: seqs.append(s)
    return seqs

if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    seqs = load_task3(cds_cache, raw, n=2000)
    print(f"Loaded {len(seqs)} sequences")

    model_cfgs = [
        ("codonbert", "<MODEL_PATH>/cLMs/CodonBERT/codonbert", False, False),
        ("codonbert_hf", "lhallee/CodonBERT", False, True),
        ("encodon-80m", "goodarzilab/encodon-80M", True, False),
    ]

    from src.models.loader import CodonModelLoader
    embeddings = {}
    for name, hf_id, is_encodon, is_rna in model_cfgs:
        print(f"Loading {name}...")
        model, tokenizer, meta = CodonModelLoader.load(name, device=DEVICE)
        emb = get_embeddings(model, tokenizer, seqs, rna=is_rna)
        embeddings[name] = emb
        print(f"  {name}: shape={emb.shape}")
        CodonModelLoader.release(model, DEVICE)

    pairs = [("codonbert", "codonbert_hf"), ("codonbert", "encodon-80m"), ("codonbert_hf", "encodon-80m")]
    results = []
    for m1, m2 in pairs:
        X, Y = embeddings[m1], embeddings[m2]
        print(f"\n{m1} vs {m2}: X.shape={X.shape}, Y.shape={Y.shape}")
        lin = linear_CKA(X, Y)
        rbf = rbf_CKA(X, Y)
        print(f"  Linear CKA = {lin:.4f}, RBF CKA = {rbf:.4f}")
        results.append({"model1": m1, "model2": m2, "linear_cka": round(lin, 4), "rbf_cka": round(rbf, 4),
                        "X_shape": list(X.shape), "Y_shape": list(Y.shape)})

    with open("results/cka_revalidation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/cka_revalidation.json")