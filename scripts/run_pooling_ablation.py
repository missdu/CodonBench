"""Pooling ablation: test whether pooling strategy explains mRNABERT's low LR→MLP gain.

Experiment 1: mRNABERT with 3 pooling strategies (mean / CLS / variant-position)
Experiment 2: CodonBERT with 2 pooling strategies (CLS / mean)

Uses cached embeddings from previous runs.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = Path("./data")
OUT_DIR = Path("./results")
SUPP_DIR = Path("./results/supplementary")
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

def eval_pooling(emb, labels, pooling_name):
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labels, test_size=0.2, random_state=42, stratify=labels)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
    lr_test = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    lr_cv = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        emb, labels, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc"
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1
    ).fit(X_tr, y_tr)
    mlp_test = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
    r = {
        "pooling": pooling_name,
        "cv_lr_mean": round(float(lr_cv.mean()), 4),
        "cv_lr_std": round(float(lr_cv.std()), 4),
        "test_lr": round(float(lr_test), 4),
        "test_mlp": round(float(mlp_test), 4),
        "lr_to_mlp_gain": round(float(mlp_test - lr_test), 4),
    }
    print(f"  {pooling_name}: CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
          f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
          f"LR→MLP={r['lr_to_mlp_gain']:+.4f}")
    return r

# ============================================================
# Experiment 1: mRNABERT pooling ablation
# ============================================================
def run_mrnabert_pooling_ablation(cds_cache, variants):
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: mRNABERT pooling ablation")
    print(f"{'='*60}")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    snapshot = "~/.cache/huggingface/hub/models--YYLY66--mRNABERT/snapshots/a1eb7df25804d23f08646e1cb996b234d7208a40"
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)

    import bert_layers as bl
    from transformers import AutoTokenizer

    model = bl.BertForMaskedLM.from_pretrained("YYLY66/mRNABERT", trust_remote_code=True)
    bert = model.bert.to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("YYLY66/mRNABERT", trust_remote_code=True)

    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, max3)]

    all_results = []

    for task_name, task_df, max_n in tasks:
        print(f"\n--- mRNABERT | {task_name} ---")
        df = balance(task_df.copy(), max_n)

        seqs, labels_list, cpos_list = [], [], []
        for _, row in df.iterrows():
            s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if s is None: continue
            seqs.append(s)
            labels_list.append(row["label"])
            cpos_list.append(int(row["cpos"]))
        labels = np.array(labels_list, dtype=int)
        print(f"  {len(seqs)} sequences")

        # Extract full hidden states (all tokens, all layers)
        # We need per-token embeddings for CLS and variant-position pooling
        all_hidden = []
        all_attn_masks = []
        all_variant_positions = []

        batch_size = 16
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            batch_cpos = cpos_list[i:i+batch_size]
            encoded = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = encoded["input_ids"].to(DEVICE)
            attn_mask = encoded["attention_mask"].to(DEVICE)

            with torch.no_grad():
                out = bert(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

            # out.hidden_states: tuple of (batch, seq_len, hidden_dim) for each layer
            # Use last layer hidden states
            if isinstance(out, tuple) and hasattr(out, 'hidden_states') and out.hidden_states is not None:
                last_hidden = out.hidden_states[-1]
            elif isinstance(out, tuple):
                last_hidden = out[0]
            else:
                last_hidden = out.last_hidden_state

            all_hidden.append(last_hidden.cpu())
            all_attn_masks.append(attn_mask.cpu())

            # Find variant position in tokenized sequence
            for j, cpos in enumerate(batch_cpos):
                ci = (cpos - 1) // 3
                s_idx = max(0, ci - CODON_CONTEXT)
                local_ci = ci - s_idx + 1  # +1 for CLS token
                if local_ci < encoded["input_ids"].shape[1]:
                    all_variant_positions.append(local_ci)
                else:
                    all_variant_positions.append(0)  # fallback to CLS

        # Stack all hidden states
        hidden = torch.cat(all_hidden, dim=0)  # (n, seq_len, 768)
        attn_masks = torch.cat(all_attn_masks, dim=0)  # (n, seq_len)

        # Pooling strategy 1: Mean pooling (current)
        print(f"  Mean pooling:")
        attn_expanded = attn_masks.unsqueeze(-1)
        mean_emb = (hidden * attn_expanded).sum(dim=1) / attn_expanded.sum(dim=1)
        r1 = eval_pooling(mean_emb.numpy(), labels, "mean")

        # Pooling strategy 2: CLS token
        print(f"  CLS pooling:")
        cls_emb = hidden[:, 0, :]
        r2 = eval_pooling(cls_emb.numpy(), labels, "cls")

        # Pooling strategy 3: Variant-position token
        print(f"  Variant-position pooling:")
        var_emb = torch.stack([hidden[i, all_variant_positions[i], :] for i in range(len(all_variant_positions))])
        r3 = eval_pooling(var_emb.numpy(), labels, "variant_pos")

        for r in [r1, r2, r3]:
            r["model"] = "mRNABERT"
            r["task"] = task_name
            all_results.append(r)

    del bert, model
    torch.cuda.empty_cache()
    return all_results

# ============================================================
# Experiment 2: CodonBERT pooling ablation
# ============================================================
def run_codonbert_pooling_ablation(cds_cache, variants):
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: CodonBERT pooling ablation")
    print(f"{'='*60}")

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader

    model, tokenizer, meta = CodonModelLoader.load("codonbert", device=DEVICE)
    if model is None:
        print("  FAILED to load CodonBERT")
        return []

    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, max3)]
    all_results = []

    for task_name, task_df, max_n in tasks:
        print(f"\n--- CodonBERT | {task_name} ---")
        df = balance(task_df.copy(), max_n)

        seqs, labels_list, cpos_list = [], [], []
        for _, row in df.iterrows():
            s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if s is None: continue
            seqs.append(s)
            labels_list.append(row["label"])
            cpos_list.append(int(row["cpos"]))
        labels = np.array(labels_list, dtype=int)
        print(f"  {len(seqs)} sequences")

        all_hidden = []
        all_attn_masks = []
        all_variant_positions = []

        batch_size = 16
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            batch_cpos = cpos_list[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = {k: v.to(DEVICE) for k, v in encoded.items()}

            with torch.no_grad():
                out = model(**input_ids, output_hidden_states=True)

            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                last_hidden = out.hidden_states[-1]
            else:
                last_hidden = out.last_hidden_state

            all_hidden.append(last_hidden.cpu())
            all_attn_masks.append(encoded["attention_mask"].cpu())

            for j, cpos in enumerate(batch_cpos):
                ci = (cpos - 1) // 3
                s_idx = max(0, ci - CODON_CONTEXT)
                local_ci = ci - s_idx + 1
                if local_ci < encoded["input_ids"].shape[1]:
                    all_variant_positions.append(local_ci)
                else:
                    all_variant_positions.append(0)

        hidden = torch.cat(all_hidden, dim=0)
        attn_masks = torch.cat(all_attn_masks, dim=0)

        # Pooling strategy 1: CLS token (current, CodonBERT default)
        print(f"  CLS pooling (default):")
        cls_emb = hidden[:, 0, :]
        r1 = eval_pooling(cls_emb.numpy(), labels, "cls")

        # Pooling strategy 2: Mean pooling
        print(f"  Mean pooling:")
        attn_expanded = attn_masks.unsqueeze(-1)
        mean_emb = (hidden * attn_expanded).sum(dim=1) / attn_expanded.sum(dim=1)
        r2 = eval_pooling(mean_emb.numpy(), labels, "mean")

        # Pooling strategy 3: Variant-position token
        print(f"  Variant-position pooling:")
        var_emb = torch.stack([hidden[i, all_variant_positions[i], :] for i in range(len(all_variant_positions))])
        r3 = eval_pooling(var_emb.numpy(), labels, "variant_pos")

        for r in [r1, r2, r3]:
            r["model"] = "CodonBERT"
            r["task"] = task_name
            all_results.append(r)

    CodonModelLoader.release(model, DEVICE)
    return all_results

def main():
    cds_cache, variants = load_data()

    results = []

    # Experiment 1
    try:
        r1 = run_mrnabert_pooling_ablation(cds_cache, variants)
        results.extend(r1)
    except Exception as e:
        print(f"mRNABERT pooling ablation FAILED: {e}")
        import traceback; traceback.print_exc()

    # Experiment 2
    try:
        r2 = run_codonbert_pooling_ablation(cds_cache, variants)
        results.extend(r2)
    except Exception as e:
        print(f"CodonBERT pooling ablation FAILED: {e}")
        import traceback; traceback.print_exc()

    # Save results
    out_file = OUT_DIR / "pooling_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("POOLING ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['model']:15s} | {r['task']:25s} | {r['pooling']:12s} | "
              f"CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
              f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
              f"LR→MLP={r['lr_to_mlp_gain']:+.4f}")

if __name__ == "__main__":
    main()