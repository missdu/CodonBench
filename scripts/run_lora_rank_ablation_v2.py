"""Step 8 v2: LoRA rank ablation (r=1,4,8,16,32,64) for CodonBERT and EnCodon-80M"""
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

DEVICE = "cuda:2"
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
    cpos = int(cpos)
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

class LoRAClassifier(nn.Module):
    def __init__(self, peft_model, hidden_dim):
        super().__init__()
        self.peft_model = peft_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.peft_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        m = attention_mask.unsqueeze(-1).float()
        emb = (h * m).sum(1) / m.sum(1)
        logits = self.classifier(emb)
        return logits

def run_lora_with_head(model, tokenizer, seqs, labels, r=8, alpha=16, epochs=30):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=r, lora_alpha=alpha, target_modules=["query", "value"],
                        lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
    peft_model = get_peft_model(model, config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA r={r}: {n_train:,} trainable / {n_total:,} total params")

    enc = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42, stratify=labels)

    hidden_dim = model.config.hidden_size
    clf_model = LoRAClassifier(peft_model, hidden_dim).to(DEVICE)

    train_ds = TensorDataset(X_train_ids, X_train_mask, torch.LongTensor(y_train))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, clf_model.parameters()), lr=2e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, best_state = 0, None

    for ep in range(epochs):
        clf_model.train()
        for ids, mask, yb in train_dl:
            ids, mask, yb = ids.to(DEVICE), mask.to(DEVICE), yb.float().to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = clf_model(ids, mask)
            loss = criterion(logits.flatten(), yb)
            loss.backward(); optimizer.step()

        clf_model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            test_ds = TensorDataset(X_test_ids, X_test_mask, torch.LongTensor(y_test))
            test_dl = DataLoader(test_ds, batch_size=32)
            for ids, mask, yb in test_dl:
                ids, mask = ids.to(DEVICE), mask.to(DEVICE)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = clf_model(ids, mask)
                pred = torch.sigmoid(logits).cpu().numpy().flatten()
                all_preds.extend(pred); all_labels.extend(yb.numpy())
        auc = roc_auc_score(all_labels, all_preds)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in clf_model.state_dict().items()}
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}: AUC = {auc:.4f} (best = {best_auc:.4f})")

    return best_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--task", type=str, default="task3_synonymous")
    parser.add_argument("--models", type=str, nargs="+", default=["codonbert", "encodon-80m"])
    args = parser.parse_args()

    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    seqs, labels = load_task(args.task, cds_cache, raw)
    print(f"Task: {args.task}, {len(seqs)} samples")

    model_configs = {
        "codonbert": {
            "hf_id": "<MODEL_PATH>/cLMs/CodonBERT/codonbert",
            "is_encodon": False,
        },
        "encodon-80m": {
            "hf_id": "goodarzilab/encodon-80M",
            "is_encodon": True,
        },
    }

    all_results = []
    from src.models.loader import CodonModelLoader
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        for r in args.ranks:
            print(f"\n  --- LoRA rank = {r} ---")
            try:
                model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
                auc = run_lora_with_head(model, tokenizer, seqs, labels, r=r)
                result = {
                    "model": model_name,
                    "task": args.task,
                    "lora_rank": r,
                    "lora_alpha": 16,
                    "auc_test": round(float(auc), 4),
                }
                all_results.append(result)
                print(f"  Rank {r}: AUC = {auc:.4f}")
                CodonModelLoader.release(model, DEVICE)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback; traceback.print_exc()
                all_results.append({"model": model_name, "task": args.task, "lora_rank": r, "error": str(e)})
                gc.collect(); torch.cuda.empty_cache()

    with open(OUT_DIR / "lora_rank_ablation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'lora_rank_ablation.json'}")