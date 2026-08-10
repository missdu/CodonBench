"""Step 3: Verify Mistral-Codon embedding quality (no NaN/Inf, norm distribution)"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json, re, time, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/embedding_quality")
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

def load_task2_data():
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
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    task2 = variants[~variants["is_syn"]].copy()
    n_pc = min(2500, int(task2["label"].sum()), int(len(task2)-task2["label"].sum()))
    df = pd.concat([task2[task2["label"]==1].sample(n_pc, random_state=42),
                    task2[task2["label"]==0].sample(n_pc, random_state=42)])
    seqs, labels = [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds[row["tx_id"]], row["cpos"])
        if s: seqs.append(s); labels.append(row["label"])
    return seqs[:500], labels[:500]

def clean_inputs(input_ids, attention_mask, token_type_ids=None):
    d = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None:
        if token_type_ids.shape[-1] != input_ids.shape[-1]:
            return d
        d["token_type_ids"] = token_type_ids
    return d

def check_model(model_name, hf_id, seqs, max_samples=500):
    print(f"\n=== Checking {model_name} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, config=config)
        model = model.to(DEVICE).eval()
    except Exception as e:
        print(f"  Failed to load: {e}")
        return {"model": model_name, "error": str(e)}

    all_embeds = []
    nan_count, inf_count, error_count = 0, 0, 0
    batch_size = 8

    for i in range(0, min(len(seqs), max_samples), batch_size):
        batch = seqs[i:i+batch_size]
        try:
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            token_type_ids = enc.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            input_ids = input_ids.clamp(0, config.vocab_size - 1)
            inputs = clean_inputs(input_ids, attention_mask, token_type_ids)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1)
            emb = emb.cpu().numpy()
            all_embeds.append(emb)
            nan_count += np.isnan(emb).sum()
            inf_count += np.isinf(emb).sum()
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"  Error at batch {i}: {e}")

    if all_embeds:
        all_embeds = np.concatenate(all_embeds, axis=0)
        norms = np.linalg.norm(all_embeds, axis=1)
        result = {
            "model": model_name,
            "n_samples": len(all_embeds),
            "nan_count": int(nan_count),
            "nan_ratio": float(nan_count / all_embeds.size),
            "inf_count": int(inf_count),
            "inf_ratio": float(inf_count / all_embeds.size),
            "error_count": error_count,
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_median": float(np.median(norms)),
            "embed_mean": float(np.nanmean(all_embeds)),
            "embed_std": float(np.nanstd(all_embeds)),
            "quality": "OK" if nan_count == 0 and inf_count == 0 and error_count == 0 else "ISSUE"
        }
    else:
        result = {"model": model_name, "n_samples": 0, "quality": "ALL_ERRORS"}

    for k, v in result.items():
        print(f"  {k}: {v}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return result

if __name__ == "__main__":
    print("Loading data...")
    seqs, labels = load_task2_data()
    print(f"Loaded {len(seqs)} sequences")

    models_to_check = [
        ("mistral-codon-117m", "Miskav/Codon-Mistral-117M"),
        ("mistral-codon-16m", "Miskav/Codon-Mistral-16M"),
        ("mistral-codon-1m", "Miskav/Codon-Mistral-1M"),
    ]

    all_results = []
    for name, hf_id in models_to_check:
        r = check_model(name, hf_id, seqs)
        all_results.append(r)

    out_path = OUT_DIR / "mistral_embedding_quality.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")