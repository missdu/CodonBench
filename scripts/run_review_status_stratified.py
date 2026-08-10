"""Step 5: ClinVar review status stratified evaluation
Check if cLM advantage holds on high-confidence (expert panel) subset"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json, re, time, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/review_status_stratified")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}

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

def get_review_tier(review_status):
    rs = str(review_status).lower()
    if "practice guideline" in rs or "reviewed by expert panel" in rs:
        return 4
    elif "reviewed by expert panel" in rs:
        return 4
    elif "no conflicts" in rs or "criteria provided" in rs:
        return 3
    elif "single submitter" in rs:
        return 2
    else:
        return 1

def load_data_with_review():
    cds = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
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
    valid["review_tier"] = valid["ReviewStatus"].apply(get_review_tier)
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    return variants, cds

def get_embeddings(model_name, hf_id, seqs, is_rna=False, is_encodon620=False):
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    try:
        if is_encodon620:
            exec(open("scripts/patch_encodon620_v2.py").read())
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, config=config)
        model = model.to(DEVICE).eval()
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        return None

    all_embeds = []
    batch_size = 8
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        if is_rna:
            batch = [s.replace("T", "U") for s in batch]
        try:
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            ids = enc["input_ids"].clamp(0, config.vocab_size - 1).to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE)
            with torch.no_grad():
                out = model(ids, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[-1]
            m = mask.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1)
            all_embeds.append(emb.cpu().numpy())
        except Exception as e:
            print(f"  Error at batch {i}: {e}")

    del model; gc.collect(); torch.cuda.empty_cache()
    if all_embeds:
        return np.concatenate(all_embeds, axis=0)
    return None

def eval_lr(X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        lr.fit(X[train_idx], y[train_idx])
        pred = lr.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y[test_idx], pred))
    return fold_aucs

MODELS = [
    ("codonbert", "~/CodonBench/cLMs/CodonBERT/codonbert", False, False),
    ("codonbert_hf", "lhallee/CodonBERT", True, False),
    ("encodon-80m", "goodarzilab/encodon-80M", False, False),
    ("encodon-620m", "goodarzilab/encodon-620M", False, True),
]

if __name__ == "__main__":
    print("Loading data with review status...")
    variants, cds_cache = load_data_with_review()

    task3 = variants[variants["is_syn"]].copy()
    print(f"\nTask 3 synonymous variants: {len(task3)}")
    print("\nReview tier distribution:")
    for tier in [4, 3, 2, 1]:
        sub = task3[task3["review_tier"] == tier]
        n_pos = int(sub["label"].sum())
        n_neg = int(len(sub) - n_pos)
        print(f"  Tier {tier}: {len(sub)} total ({n_pos} pathogenic, {n_neg} benign)")

    all_results = []
    for tier in [4, 3, 2, 1]:
        sub = task3[task3["review_tier"] == tier].copy()
        n_pos = int(sub["label"].sum())
        n_neg = int(len(sub) - n_pos)
        if n_pos < 20 or n_neg < 20:
            print(f"\n=== Tier {tier}: SKIPPED (too few: {n_pos} pos, {n_neg} neg) ===")
            continue
        n_pc = min(n_neg, n_pos, 2500)
        df = pd.concat([sub[sub["label"]==1].sample(n_pc, random_state=42),
                        sub[sub["label"]==0].sample(n_pc, random_state=42)])
        seqs, labels = [], []
        for _, row in df.iterrows():
            s = build_codon_str(cds_cache[row["tx_id"]], row["cpos"])
            if s: seqs.append(s); labels.append(row["label"])
        labels = np.array(labels)
        print(f"\n=== Tier {tier}: {len(seqs)} samples ({labels.sum()} pos, {(1-labels).sum()} neg) ===")

        for model_name, hf_id, is_rna, is_encodon620 in MODELS:
            print(f"  Evaluating {model_name}...")
            hf_path = os.path.expanduser(hf_id) if hf_id.startswith("~") else hf_id
            X = get_embeddings(model_name, hf_path, seqs, is_rna, is_encodon620)
            if X is None:
                continue
            fold_aucs = eval_lr(X, labels)
            result = {
                "review_tier": tier,
                "model": model_name,
                "n_samples": len(X),
                "n_pathogenic": int(labels.sum()),
                "n_benign": int((1-labels).sum()),
                "auc_mean": float(np.mean(fold_aucs)),
                "auc_std": float(np.std(fold_aucs)),
                "auc_folds": [float(x) for x in fold_aucs],
            }
            all_results.append(result)
            print(f"    AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    with open(OUT_DIR / "review_status_stratified.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'review_status_stratified.json'}")