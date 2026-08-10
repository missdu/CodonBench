"""Step 8: LoRA rank ablation (r=1,4,8,16,32,64) for CodonBERT and EnCodon-620M"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json, re, time, gc, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/lora_rank_ablation")
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

def load_task(task_name, cds_cache, raw_data):
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
    if task_name == "task3_synonymous":
        df = variants[variants["is_syn"]].copy()
        max_n = 2840
    else:
        df = variants[~variants["is_syn"]].copy()
        max_n = 5000
    n_pc = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
    df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                    df[df["label"]==0].sample(n_pc, random_state=42)])
    seqs, labels = [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds_cache[row["tx_id"]], row["cpos"])
        if s: seqs.append(s); labels.append(row["label"])
    return seqs, np.array(labels)

def inject_lora(model, r=8, alpha=16):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=["query", "value"],
                        lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, config)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={r}: {n_train:,} trainable / {n_total:,} total params")
    return model

def run_lora(model, tokenizer, seqs, labels, r=8, alpha=16, epochs=30):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("codonbert_placeholder", trust_remote_code=True)
    model = inject_lora(model, r, alpha)
    enc = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42, stratify=labels)

    train_ds = TensorDataset(X_train_ids, X_train_mask, torch.LongTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for ep in range(epochs):
        model.train()
        for ids, mask, yb in train_dl:
            ids, mask, yb = ids.to(DEVICE), mask.to(DEVICE), yb.float().to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(ids, attention_mask=mask, output_hidden_states=True)
                h = out.hidden_states[-1]
                m = mask.unsqueeze(-1).float()
                emb = (h * m).sum(1) / m.sum(1)
                logits = model.modules().get("classifier", nn.Linear(emb.shape[-1], 1).to(DEVICE))(emb.detach())
            loss = criterion(logits.flatten(), yb)
            loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            test_ds = TensorDataset(X_test_ids, X_test_mask, torch.LongTensor(y_test))
            test_dl = DataLoader(test_ds, batch_size=32)
            for ids, mask, yb in test_dl:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                out = model(ids, attention_mask=mask, output_hidden_states=True)
                h = out.hidden_states[-1]
                m = mask.unsqueeze(-1).float()
                emb = (h * m).sum(1) / m.sum(1)
                pred = torch.sigmoid(emb.mean(dim=1)).cpu().numpy()
                all_preds.extend(pred); all_labels.extend(yb.numpy())
        auc = roc_auc_score(all_labels, all_preds)
        if auc > best_auc:
            best_auc = auc
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}: AUC = {auc:.4f} (best = {best_auc:.4f})")

    return best_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--task", type=str, default="task3_synonymous")
    args = parser.parse_args()

    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    seqs, labels = load_task(args.task, cds_cache, raw)
    print(f"Task: {args.task}, {len(seqs)} samples")

    model_configs = [
        ("codonbert", os.path.expanduser("~/CodonBench/cLMs/CodonBERT/codonbert"), False),
    ]

    all_results = []
    for model_name, hf_id, is_encodon620 in model_configs:
        print(f"\n=== {model_name} ===")
        for r in args.ranks:
            print(f"\n  --- LoRA rank = {r} ---")
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                if is_encodon620:
                    exec(open("scripts/patch_encodon620_v2.py").read())
                tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
                config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
                model = AutoModel.from_pretrained(hf_id, trust_remote_code=True, config=config)
                model = model.to(DEVICE)
                auc = run_lora(model, tokenizer, seqs, labels, r=r)
                result = {
                    "model": model_name,
                    "task": args.task,
                    "lora_rank": r,
                    "lora_alpha": 16,
                    "auc_test": float(auc),
                }
                all_results.append(result)
                print(f"  Rank {r}: AUC = {auc:.4f}")
                del model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Error: {e}")
                all_results.append({"model": model_name, "task": args.task, "lora_rank": r, "error": str(e)})

    with open(OUT_DIR / "lora_rank_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'lora_rank_ablation.json'}")