"""P0-3: mRNABERT evaluation with MosaicBERT custom loading.

mRNABERT uses MosaicBERT architecture (fused QKV, gated MLP).
We load it by importing the custom code from the HF cache.
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

DEVICE = "cuda:0"  # Will be overridden by CUDA_VISIBLE_DEVICES
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

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

def load_mrnabert_model():
    """Load mRNABERT model using custom code from HF cache.
    
    mRNABERT uses MosaicBERT architecture with relative imports in bert_layers.py.
    We fix this by: (1) creating __init__.py in the snapshot dir, 
    (2) patching relative imports to absolute, (3) adding dir to sys.path.
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download

    model_id = "YYLY66/mRNABERT"

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    snapshot_dir = Path(hf_hub_download(model_id, "config.json")).parent
    bert_layers_path = snapshot_dir / "bert_layers.py"

    if not bert_layers_path.exists():
        raise FileNotFoundError(f"bert_layers.py not found at {bert_layers_path}")

    # Create __init__.py to make snapshot dir a proper package
    init_path = snapshot_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("# mRNABERT package\n")

    # Patch relative imports to absolute
    with open(bert_layers_path) as f:
        content = f.read()
    patched = content.replace("from .bert_padding import", "from bert_padding import")
    patched = patched.replace("from .flash_attn_triton import", "from flash_attn_triton import")
    if patched != content:
        with open(bert_layers_path, "w") as f:
            f.write(patched)

    # Add snapshot dir to sys.path and import
    snapshot_str = str(snapshot_dir)
    if snapshot_str not in sys.path:
        sys.path.insert(0, snapshot_str)

    import bert_layers as bl
    model = bl.BertForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
    bert_model = model.bert

    return bert_model, tok

def extract_embeddings(model, tokenizer, seqs, device=DEVICE, batch_size=16):
    all_embs = []
    t0 = time.time()
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        try:
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
            with torch.no_grad():
                out = model(**{k: v.to(device) for k, v in encoded.items()})
            # mRNABERT returns tuple, not dict; first element is last_hidden_state
            if isinstance(out, tuple):
                hidden = out[0]
            else:
                hidden = out.last_hidden_state
            # Use attention-masked mean pooling (MosaicBERT uses unpadding, CLS may not be reliable)
            attn_mask = encoded["attention_mask"].unsqueeze(-1).to(device)
            masked_hidden = hidden * attn_mask
            emb = (masked_hidden.sum(dim=1) / attn_mask.sum(dim=1)).cpu().numpy()
            all_embs.append(emb)
        except Exception as e:
            print(f"  Batch error at {i}: {e}")
            for s in batch:
                try:
                    enc = tokenizer([s], padding=True, truncation=True,
                                    max_length=512, return_tensors="pt")
                    with torch.no_grad():
                        out = model(**{k: v.to(device) for k, v in enc.items()})
                    if isinstance(out, tuple):
                        hidden = out[0]
                    else:
                        hidden = out.last_hidden_state
                    attn_mask = enc["attention_mask"].unsqueeze(-1).to(device)
                    masked_hidden = hidden * attn_mask
                    emb = (masked_hidden.sum(dim=1) / attn_mask.sum(dim=1)).cpu().numpy()
                    all_embs.append(emb)
                except:
                    pass
        if (i // batch_size) % 50 == 0:
            print(f"  Batch {i//batch_size+1}/{(len(seqs)-1)//batch_size+1}")
    emb = np.vstack(all_embs)
    elapsed = time.time() - t0
    print(f"  Embeddings: {emb.shape}, {elapsed:.1f}s")
    return emb

def run_mrnabert(task_name, task_df, max_n, cds_cache):
    emb_path = OUT_DIR / f"mrnabert_{task_name}_emb.npy"
    lab_path = OUT_DIR / f"mrnabert_{task_name}_labels.npy"

    if emb_path.exists() and lab_path.exists():
        print(f"  Loading cached embeddings from {emb_path}")
        emb = np.load(emb_path)
        labs = np.load(lab_path)
    else:
        print(f"  Loading mRNABERT model...")
        model, tokenizer = load_mrnabert_model()
        model = model.to(DEVICE).eval()
        print(f"  mRNABERT loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        df = balance(task_df.copy(), max_n)
        seqs, labs_list = [], []
        for _, row in df.iterrows():
            s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if s is None: continue
            seqs.append(s)
            labs_list.append(row["label"])
        labs = np.array(labs_list, dtype=int)
        print(f"  {task_name}: {len(seqs)} sequences")

        emb = extract_embeddings(model, tokenizer, seqs)
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
        "model": "mRNABERT",
        "task": task_name,
        "model_type": "cLM (MosaicBERT, codon tokenizer)",
        "n": int(len(labs)),
        "cv_lr_mean": round(float(lr_cv.mean()), 4),
        "cv_lr_std": round(float(lr_cv.std()), 4),
        "test_lr": round(float(lr_test), 4),
        "test_mlp": round(float(mlp_test), 4),
        "lr_to_mlp_gain": round(float(mlp_test - lr_test), 4),
    }
    print(f"  CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
          f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
          f"LR→MLP gain={r['lr_to_mlp_gain']:+.4f}")
    return r

def main():
    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    tasks = [
        ("task2_missense", t2, 5000),
        ("task3_synonymous", t3, max3),
    ]

    res_file = OUT_DIR / "mrnabert_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for task_name, task_df, max_n in tasks:
        if ("mRNABERT", task_name) in done_keys:
            print(f"  SKIP mRNABERT {task_name} (already done)")
            continue
        print(f"\n{'='*60}\nmRNABERT | {task_name}\n{'='*60}")
        r = run_mrnabert(task_name, task_df, max_n, cds_cache)
        if r:
            existing.append(r)
            with open(res_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nmRNABERT RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:15s} | {r['task']:25s} | "
              f"CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
              f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
              f"LR→MLP={r['lr_to_mlp_gain']:+.4f}")

if __name__ == "__main__":
    main()
