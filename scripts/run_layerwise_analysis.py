"""Experiment 4: Layer-wise LLR and MLP analysis for CodonBERT-HF.

Purpose: Prove that "MLM loss insensitivity ≠ information absence"
by showing LLR AUC ≈ 0.5 at all layers but MLP AUC increases with depth.
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results")
CODON_CONTEXT = 16

def dna_to_rna_codons(seq):
    return [seq[i:i+3].replace("T","U") for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3)), m.group(4), m.group(5)) if m else (None, None, None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_rna_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def build_mutant_codon_str(cds, cpos, ref_allele, alt_allele, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    cds_list = list(cds)
    pos0 = cpos - 1
    if cds_list[pos0] != ref_allele: return None
    cds_list[pos0] = alt_allele
    mutant_cds = "".join(cds_list)
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(mutant_cds) // 3, ci + ctx + 1)
    c = dna_to_rna_codons(mutant_cds[s*3:e*3])
    return " ".join(c) if c else None

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
    valid["ref_allele"] = parsed.apply(lambda x: x[2])
    valid["alt_allele"] = parsed.apply(lambda x: x[3])
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

def get_layer_embeddings(model, tokenizer, seqs, layer_idx, device=DEVICE, batch_size=16):
    """Extract CLS embeddings from a specific layer."""
    all_embs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            layer_emb = out.hidden_states[layer_idx][:, 0, :].cpu().numpy()
        else:
            layer_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(layer_emb)
    return np.vstack(all_embs)

def compute_llr_at_layer(model, tokenizer, wt_seq, mut_seq, layer_idx, device=DEVICE):
    """Compute log-likelihood ratio at a specific layer using the MLM head."""
    def get_log_prob(seq):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
        return -out.loss.item() * inputs["input_ids"].shape[1]
    try:
        wt_lp = get_log_prob(wt_seq)
        mut_lp = get_log_prob(mut_seq)
        return mut_lp - wt_lp
    except:
        return None

def main():
    cds_cache, variants = load_data()

    # Focus on synonymous variants (the key channel)
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    # Also missense for comparison
    t2 = variants[~variants["is_syn"]].copy()

    print(f"\n{'='*60}")
    print("EXPERIMENT 4: Layer-wise LLR and MLP analysis (CodonBERT-HF)")
    print(f"{'='*60}")

    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import _fix_token_type_ids

    model, tokenizer, meta = CodonModelLoader.load("codonbert_hf", device=DEVICE)
    if model is None:
        print("FAILED to load CodonBERT-HF")
        return

    n_layers = model.config.num_hidden_layers
    print(f"Model: CodonBERT-HF, {n_layers} layers")

    results = []

    # Part A: Layer-wise MLP probing on SynPath
    print(f"\n--- Part A: Layer-wise MLP probing (SynPath) ---")
    df_syn = balance(t3.copy(), max3)
    seqs_syn, labels_syn = [], []
    for _, row in df_syn.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s is None: continue
        seqs_syn.append(s)
        labels_syn.append(row["label"])
    labels_syn = np.array(labels_syn, dtype=int)
    print(f"  SynPath: {len(seqs_syn)} sequences")

    for layer_idx in range(n_layers + 1):  # 0=embedding, 1..n=transformer layers
        print(f"  Layer {layer_idx}/{n_layers}...", end=" ", flush=True)
        emb = get_layer_embeddings(model, tokenizer, seqs_syn, layer_idx)

        X_tr, X_te, y_tr, y_te = train_test_split(emb, labels_syn, test_size=0.2, random_state=42, stratify=labels_syn)

        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
        lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                           early_stopping=True, validation_fraction=0.1).fit(X_tr, y_tr)
        mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])

        r = {
            "model": "CodonBERT-HF",
            "task": "task3_synonymous",
            "layer": layer_idx,
            "lr_auc": round(float(lr_auc), 4),
            "mlp_auc": round(float(mlp_auc), 4),
            "lr_to_mlp": round(float(mlp_auc - lr_auc), 4),
        }
        results.append(r)
        print(f"LR={lr_auc:.4f} | MLP={mlp_auc:.4f} | Δ={mlp_auc-lr_auc:+.4f}")

    # Part B: Layer-wise MLP probing on MisPath
    print(f"\n--- Part B: Layer-wise MLP probing (MisPath) ---")
    df_mis = balance(t2.copy(), 5000)
    seqs_mis, labels_mis = [], []
    for _, row in df_mis.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s is None: continue
        seqs_mis.append(s)
        labels_mis.append(row["label"])
    labels_mis = np.array(labels_mis, dtype=int)
    print(f"  MisPath: {len(seqs_mis)} sequences")

    for layer_idx in range(n_layers + 1):
        print(f"  Layer {layer_idx}/{n_layers}...", end=" ", flush=True)
        emb = get_layer_embeddings(model, tokenizer, seqs_mis, layer_idx)

        X_tr, X_te, y_tr, y_te = train_test_split(emb, labels_mis, test_size=0.2, random_state=42, stratify=labels_mis)

        lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
        lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                           early_stopping=True, validation_fraction=0.1).fit(X_tr, y_tr)
        mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])

        r = {
            "model": "CodonBERT-HF",
            "task": "task2_missense",
            "layer": layer_idx,
            "lr_auc": round(float(lr_auc), 4),
            "mlp_auc": round(float(mlp_auc), 4),
            "lr_to_mlp": round(float(mlp_auc - lr_auc), 4),
        }
        results.append(r)
        print(f"LR={lr_auc:.4f} | MLP={mlp_auc:.4f} | Δ={mlp_auc-lr_auc:+.4f}")

    # Part C: Zero-shot LLR (same across all layers — MLM loss is layer-agnostic)
    print(f"\n--- Part C: Zero-shot LLR (SynPath, n=500 subset) ---")
    df_llr = balance(t3.copy(), 500)  # Small subset for speed
    llrs_syn, labels_llr_syn = [], []
    for _, row in df_llr.iterrows():
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        ref = row["ref_allele"]
        alt = row["alt_allele"]
        wt_seq = build_codon_str(cds, cpos)
        mut_seq = build_mutant_codon_str(cds, cpos, ref, alt)
        if wt_seq is None or mut_seq is None: continue
        llr = compute_llr_at_layer(model, tokenizer, wt_seq, mut_seq, n_layers)
        if llr is not None:
            llrs_syn.append(llr)
            labels_llr_syn.append(row["label"])
    llrs_syn = np.array(llrs_syn)
    labels_llr_syn = np.array(labels_llr_syn)
    if len(llrs_syn) > 100 and len(set(labels_llr_syn)) == 2:
        llr_auc_syn = roc_auc_score(labels_llr_syn, -llrs_syn)
        print(f"  LLR AUC (SynPath): {llr_auc_syn:.4f} (n={len(llrs_syn)})")
        results.append({"model": "CodonBERT-HF", "task": "task3_synonymous", "layer": "llr",
                        "lr_auc": round(float(llr_auc_syn), 4), "mlp_auc": None, "lr_to_mlp": None})
    else:
        print(f"  Too few LLR samples: {len(llrs_syn)}")

    # Part D: Zero-shot LLR on MisPath
    print(f"\n--- Part D: Zero-shot LLR (MisPath, n=500 subset) ---")
    df_llr_mis = balance(t2.copy(), 500)
    llrs_mis, labels_llr_mis = [], [], 
    for _, row in df_llr_mis.iterrows():
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        ref = row["ref_allele"]
        alt = row["alt_allele"]
        wt_seq = build_codon_str(cds, cpos)
        mut_seq = build_mutant_codon_str(cds, cpos, ref, alt)
        if wt_seq is None or mut_seq is None: continue
        llr = compute_llr_at_layer(model, tokenizer, wt_seq, mut_seq, n_layers)
        if llr is not None:
            llrs_mis.append(llr)
            labels_llr_mis.append(row["label"])
    llrs_mis = np.array(llrs_mis)
    labels_llr_mis = np.array(labels_llr_mis)
    if len(llrs_mis) > 100 and len(set(labels_llr_mis)) == 2:
        llr_auc_mis = roc_auc_score(labels_llr_mis, -llrs_mis)
        print(f"  LLR AUC (MisPath): {llr_auc_mis:.4f} (n={len(llrs_mis)})")
        results.append({"model": "CodonBERT-HF", "task": "task2_missense", "layer": "llr",
                        "lr_auc": round(float(llr_auc_mis), 4), "mlp_auc": None, "lr_to_mlp": None})

    CodonModelLoader.release(model, DEVICE)

    # Save
    out_file = OUT_DIR / "layerwise_analysis_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("LAYER-WISE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r.get("mlp_auc") is not None:
            print(f"  {r['task']:25s} | Layer {str(r['layer']):>3s} | "
                  f"LR={r['lr_auc']:.4f} | MLP={r['mlp_auc']:.4f} | Δ={r['lr_to_mlp']:+.4f}")
        else:
            print(f"  {r['task']:25s} | LLR     | AUC={r['lr_auc']:.4f}")

if __name__ == "__main__":
    main()