"""LoRA rank ablation for EnCodon-80M using manual LoRA injection (not PEFT)."""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json, re, time, gc
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

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.r = r
        self.alpha = alpha
        d_out, d_in = original_layer.weight.shape
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.scaling = alpha / r

    def forward(self, x):
        orig_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return orig_out + lora_out

def inject_lora_into_encodon(model, r=8, alpha=16):
    lora_modules = []
    layers = model.bert.encoder.layer
    for i, layer in enumerate(layers):
        sa = layer.attention.self_attention
        for name in ["query", "value"]:
            orig = getattr(sa, name)
            lora_layer = LoRALayer(orig, r=r, alpha=alpha)
            setattr(sa, name, lora_layer)
            lora_modules.append(lora_layer)
    return lora_modules

def get_pooled_hidden(model, input_ids, attention_mask, token_type_ids=None):
    forward_params = set(model.forward.__code__.co_varnames) if hasattr(model, 'forward') else set()
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if token_type_ids is not None and "token_type_ids" in forward_params:
        kwargs["token_type_ids"] = token_type_ids
    if hasattr(model, 'bert'):
        outputs = model.bert(**kwargs)
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0]
    else:
        outputs = model(**kwargs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
    m = attention_mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1)

def run_lora_encodon(model, tokenizer, seqs, labels, r=8, alpha=16, epochs=30):
    lora_modules = inject_lora_into_encodon(model, r=r, alpha=alpha)
    lora_params = []
    for lm in lora_modules:
        lora_params.extend(list(lm.lora_A.parameters()))
        lora_params.extend(list(lm.lora_B.parameters()))
    n_lora = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={r}: {n_lora:,} trainable / {n_total:,} total params")

    enc = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc.get("token_type_ids", None)
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids, attention_mask, labels, test_size=0.2, random_state=42, stratify=labels)
    if token_type_ids is not None:
        X_train_tt, X_test_tt, _, _ = train_test_split(
            token_type_ids, labels, test_size=0.2, random_state=42, stratify=labels)
    else:
        X_train_tt, X_test_tt = None, None

    hidden_size = model.config.hidden_size
    classifier = nn.Linear(hidden_size, 2).to(DEVICE)
    all_params = lora_params + list(classifier.parameters())
    optimizer = AdamW(all_params, lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    batch_size = 16
    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        classifier.train()
        perm = torch.randperm(len(y_train))
        for start in range(0, len(y_train), batch_size):
            idx = perm[start:start+batch_size]
            b_ids = X_train_ids[idx].to(DEVICE)
            b_mask = X_train_mask[idx].to(DEVICE)
            b_tt = X_train_tt[idx].to(DEVICE) if X_train_tt is not None else None
            b_labels = torch.tensor(y_train[idx], dtype=torch.long).to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                emb = get_pooled_hidden(model, b_ids, b_mask, b_tt)
                logits = classifier(emb)
                loss = criterion(logits, b_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for start in range(0, len(y_test), batch_size):
                b_ids = X_test_ids[start:start+batch_size].to(DEVICE)
                b_mask = X_test_mask[start:start+batch_size].to(DEVICE)
                b_tt = X_test_tt[start:start+batch_size].to(DEVICE) if X_test_tt is not None else None
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    emb = get_pooled_hidden(model, b_ids, b_mask, b_tt)
                    logits = classifier(emb)
                all_preds.extend(logits[:, 1].sigmoid().cpu().numpy())
                all_labels.extend(y_test[start:start+batch_size])
        auc = roc_auc_score(all_labels, all_preds)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_state['classifier'] = {k: v.clone() for k, v in classifier.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: AUC = {auc:.4f} (best = {best_auc:.4f})")

    print(f"  Rank {r}: AUC = {best_auc:.4f}")
    return best_auc

if __name__ == "__main__":
    print("Loading data...")
    import gzip
    raw = pd.read_csv(gzip.open(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz"), sep="\t", low_memory=False)
    with open(DATA_DIR / "task2_clinvar" / "cds_sequences.json") as f:
        cds_cache = json.load(f)
    seqs, labels = load_task("task3_synonymous", cds_cache, raw)
    print(f"Task: task3_synonymous, {len(seqs)} samples")

    from src.models.loader import CodonModelLoader
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    ranks = [1, 4, 8, 16, 32, 64]
    results = []

    for r in ranks:
        print(f"\n  --- LoRA rank = {r} ---")
        try:
            model, tokenizer, meta = CodonModelLoader.load("encodon-80m", device=DEVICE)
            auc = run_lora_encodon(model, tokenizer, seqs, labels, r=r)
            results.append({"model": "encodon-80m", "task": "task3_synonymous", "lora_rank": r, "lora_alpha": 16, "auc_test": round(float(auc), 4)})
            CodonModelLoader.release(model, DEVICE)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()
            results.append({"model": "encodon-80m", "task": "task3_synonymous", "lora_rank": r, "error": str(e)})
            gc.collect(); torch.cuda.empty_cache()

    out_path = OUT_DIR / "lora_rank_ablation_encodon80m.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")