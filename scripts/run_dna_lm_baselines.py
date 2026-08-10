import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/dna_lm_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

def build_nucleotide_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    start_nt = start_codon * 3
    end_nt = end_codon * 3
    return cds_seq[start_nt:end_nt]

def load_dna_model(model_key):
    configs = {
        "dnabert2": {
            "hf_id": "zhihan1996/DNABERT-2-117M",
            "params_M": 117,
            "type": "dnabert2",
        },
        "nt-500m": {
            "hf_id": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
            "params_M": 500,
            "type": "nucleotide_transformer",
        },
        "nt-50m": {
            "hf_id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
            "params_M": 50,
            "type": "nucleotide_transformer",
        },
    }
    cfg = configs[model_key]
    print(f"  Loading {model_key} ({cfg['hf_id']})...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    
    if model_key == "dnabert2":
        import sys
        cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/7bce263b15377fc15361f52cfab88f8b586abda0")
        if cache_dir not in sys.path:
            sys.path.insert(0, cache_dir)
        from configuration_bert import BertConfig as DNABert2Config
        from bert_layers import BertModel as DNABert2Model
        config = DNABert2Config.from_pretrained(cfg["hf_id"], trust_remote_code=True)
        model = DNABert2Model(config).from_pretrained(cfg["hf_id"], trust_remote_code=True).to(DEVICE)
    elif model_key.startswith("nt-"):
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(cfg["hf_id"], trust_remote_code=True).to(DEVICE)
    else:
        model = AutoModel.from_pretrained(cfg["hf_id"], trust_remote_code=True).to(DEVICE)
    
    model.eval()
    print(f"  Loaded: {model.__class__.__name__}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer, cfg

def extract_dna_embeddings(model, tokenizer, sequences, model_type, batch_size=8, max_length=512):
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        if model_type == "dnabert2":
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        else:
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                embs = outputs.hidden_states[-1][:, 0, :]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embs = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                embs = outputs.last_hidden_state[:, 0, :]
            else:
                embs = outputs[0][:, 0, :]
        all_embs.append(embs.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"    Batch {i//batch_size+1}/{(len(sequences)-1)//batch_size+1}")
    
    return np.vstack(all_embs)

def evaluate_dna_model(model_key, task_name, variants_df, cds_cache, max_samples=5000):
    model, tokenizer, cfg = load_dna_model(model_key)
    
    df = variants_df.copy()
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)
    
    sequences = []
    labels = []
    skipped = 0
    
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            skipped += 1
            continue
        nt_seq = build_nucleotide_context(cds_seq, cpos)
        if nt_seq is None:
            skipped += 1
            continue
        sequences.append(nt_seq)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(sequences)} sequences (skipped {skipped}), P={int(labels.sum())} B={int(len(labels)-labels.sum())}")
    
    if len(sequences) < 100:
        del model
        torch.cuda.empty_cache()
        return {"model": model_key, "task": task_name, "success": False, "error": f"Too few: {len(sequences)}"}
    
    t0 = time.time()
    embeddings = extract_dna_embeddings(model, tokenizer, sequences, cfg["type"], batch_size=8)
    emb_time = time.time() - t0
    print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    t0 = time.time()
    aucs = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")
    cv_time = time.time() - t0
    
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    
    t_stat, p_val = stats.ttest_1samp(aucs, 0.5)
    neg_log10_p = -np.log10(p_val) if p_val > 0 else float("inf")
    
    result = {
        "model": model_key, "task": task_name,
        "model_type": cfg["type"], "params_M": cfg["params_M"],
        "success": True, "n_samples": len(sequences),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "ROC-AUC_mean": round(float(mean_auc), 4),
        "ROC-AUC_std": round(float(std_auc), 4),
        "ROC-AUC_folds": [round(float(a), 4) for a in aucs],
        "neg_log10_p": round(float(neg_log10_p), 4) if not np.isnan(neg_log10_p) else None,
        "emb_time_s": round(emb_time, 1), "cv_time_s": round(cv_time, 1),
        "context_codons": CODON_CONTEXT,
        "input_type": "nucleotide",
    }
    
    result_file = OUT_DIR / f"{model_key}_{task_name}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Result: ROC-AUC={mean_auc:.4f}+/-{std_auc:.4f}, -log10(p)={result['neg_log10_p']}")
    return result


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")
    
    print("Loading FULL ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    print(f"  SNVs: {len(snv):,}")
    
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
    valid["cref"] = parsed.apply(lambda x: x[2])
    valid["calt"] = parsed.apply(lambda x: x[3])
    
    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()
    
    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))
    
    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()
    print(f"  Task2 (missense): {len(task2_variants):,}")
    print(f"  Task3 (synonymous): {len(task3_variants):,}")
    
    dna_models = ["dnabert2", "nt-50m", "nt-500m"]
    
    n_task3_patho = int((task3_variants["label"]==1).sum())
    n_task3_benign = int((task3_variants["label"]==0).sum())
    task3_max = min(n_task3_patho * 2, n_task3_patho + n_task3_benign)
    
    tasks = [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]
    
    all_results = []
    for model_key in dna_models:
        for task_name, task_df, max_samples in tasks:
            if len(task_df) < 100:
                continue
            print(f"\n{'='*60}")
            print(f"Evaluating {model_key} on {task_name} (max={max_samples})")
            print(f"{'='*60}")
            try:
                result = evaluate_dna_model(
                    model_key, task_name, task_df, cds_cache,
                    max_samples=max_samples
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({"model": model_key, "task": task_name, "success": False, "error": str(e)})
    
    summary_file = OUT_DIR / "dna_lm_baselines_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"DNA LM Baseline evaluation complete!")
    print(f"{'='*60}")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:15s} | {r['task']:25s} | AUC={r['ROC-AUC_mean']:.4f}+/-{r['ROC-AUC_std']:.4f} | n={r['n_samples']} | -log10p={r.get('neg_log10_p','N/A')}")
        else:
            print(f"  {r['model']:15s} | {r.get('task','?'):25s} | FAILED: {str(r.get('error','unknown'))[:80]}")

if __name__ == "__main__":
    main()