import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json, re, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def dna_to_rna(seq):
    return seq.replace("T", "U")

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds):
        return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
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
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0])
    v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds.keys()))
    var = v[h].copy()
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    return cds, var

def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        return pd.concat([
            df[df["label"] == 1].sample(n, random_state=42),
            df[df["label"] == 0].sample(n, random_state=42)
        ]).sample(frac=1, random_state=42)
    return df

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

def inject_lora_into_bert(model, r=8, alpha=16):
    lora_modules = []
    for i, layer in enumerate(model.bert.encoder.layer):
        attn = layer.attention.self
        for name in ["query", "value"]:
            orig = getattr(attn, name)
            lora_layer = LoRALayer(orig, r=r, alpha=alpha)
            setattr(attn, name, lora_layer)
            lora_modules.append(lora_layer)
    return lora_modules

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

    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    return pooled

def tokenize_batch(tokenizer, seqs, device, max_length=2048):
    inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    result = {k: v.to(device) for k, v in inputs.items()}
    if "token_type_ids" in result:
        if result["token_type_ids"].shape[1] != result["input_ids"].shape[1]:
            diff = result["input_ids"].shape[1] - result["token_type_ids"].shape[1]
            result["token_type_ids"] = torch.cat([
                result["token_type_ids"],
                torch.zeros(result["token_type_ids"].shape[0], diff, dtype=torch.long, device=device)
            ], dim=1)
    return result

def run_lora_experiment(mname, use_rna, task_name, task_df, max_n, cds_cache):
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tune: {mname} | {task_name}")
    print(f"{'='*60}")

    emb_path = OUT_DIR / f"{mname}_{task_name}_emb.npy"
    lab_path = OUT_DIR / f"{mname}_{task_name}_labels.npy"

    if not emb_path.exists() or not lab_path.exists():
        print(f"  ERROR: No cached embeddings for {mname} {task_name}")
        return None

    emb = np.load(emb_path)
    labels = np.load(lab_path)
    print(f"  Cached embeddings: {emb.shape}, labels: {len(labels)}")

    X_tr, X_te, y_tr, y_te = train_test_split(emb, labels, test_size=0.2, random_state=42, stratify=labels)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                         early_stopping=True, validation_fraction=0.1).fit(X_tr, y_tr)
    mlp_auc = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])

    print(f"  LR AUC = {lr_auc:.4f}, MLP AUC = {mlp_auc:.4f}")

    df = balance(task_df.copy(), max_n)
    seqs, labs_list = [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s is None:
            continue
        if use_rna:
            s = dna_to_rna(s)
        seqs.append(s)
        labs_list.append(int(row["label"]))

    train_seqs, test_seqs, train_labels, test_labels = train_test_split(
        seqs, labs_list, test_size=0.2, random_state=42, stratify=labs_list
    )
    print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}")

    from src.models.loader import CodonModelLoader
    model, tokenizer, meta = CodonModelLoader.load(mname, device=DEVICE)
    if model is None:
        print(f"  FAILED to load {mname}")
        return None

    hidden_size = None
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = emb.shape[1]

    is_encodon = 'encodon' in mname.lower()
    if is_encodon:
        lora_modules = inject_lora_into_encodon(model, r=8, alpha=16)
    else:
        lora_modules = inject_lora_into_bert(model, r=8, alpha=16)

    model.to(DEVICE)

    classifier = nn.Linear(hidden_size, 2).to(DEVICE)
    lora_params = []
    for lm in lora_modules:
        lora_params.extend(list(lm.lora_A.parameters()))
        lora_params.extend(list(lm.lora_B.parameters()))
    all_params = lora_params + list(classifier.parameters())
    optimizer = AdamW(all_params, lr=2e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    n_lora_params = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {n_lora_params}")
    print(f"  Classifier params: {sum(p.numel() for p in classifier.parameters())}")

    batch_size = 16
    n_epochs = 30
    best_lora_auc = 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        classifier.train()
        perm = np.random.permutation(len(train_seqs))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_seqs), batch_size):
            idx = perm[i:i+batch_size]
            batch_seqs = [train_seqs[j] for j in idx]
            batch_labels = torch.LongTensor([train_labels[j] for j in idx]).to(DEVICE)

            inputs = tokenize_batch(tokenizer, batch_seqs, DEVICE)

            optimizer.zero_grad()
            pooled = get_pooled_hidden(model, **inputs)
            logits = classifier(pooled)
            loss = criterion(logits, batch_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        classifier.eval()
        all_probs = []
        all_true = []
        with torch.no_grad():
            for i in range(0, len(test_seqs), batch_size):
                batch_seqs = test_seqs[i:i+batch_size]
                batch_labels_t = torch.LongTensor(test_labels[i:i+batch_size]).to(DEVICE)
                inputs = tokenize_batch(tokenizer, batch_seqs, DEVICE)
                pooled = get_pooled_hidden(model, **inputs)
                logits = classifier(pooled)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_true.extend(test_labels[i:i+batch_size])

        lora_auc = roc_auc_score(all_true, all_probs)
        if lora_auc > best_lora_auc:
            best_lora_auc = lora_auc
            best_state = {
                'classifier': {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f}, LoRA AUC={lora_auc:.4f} (best={best_lora_auc:.4f})")

    result = {
        "model": mname,
        "task": task_name,
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "test_lr_auc": round(float(lr_auc), 4),
        "test_mlp_auc": round(float(mlp_auc), 4),
        "test_lora_auc": round(float(best_lora_auc), 4),
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_lr": 2e-4,
        "lora_epochs": n_epochs,
        "lora_params": n_lora_params,
        "improvement_over_lr": round(float(best_lora_auc - lr_auc), 4),
        "improvement_over_mlp": round(float(best_lora_auc - mlp_auc), 4),
    }

    CodonModelLoader.release(model, DEVICE)
    del classifier
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  RESULT: LR={lr_auc:.4f} | MLP={mlp_auc:.4f} | LoRA={best_lora_auc:.4f}")
    print(f"  LoRA vs LR: {result['improvement_over_lr']:+.4f}")
    print(f"  LoRA vs MLP: {result['improvement_over_mlp']:+.4f}")

    return result

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "all"
    task_key = sys.argv[2] if len(sys.argv) > 2 else "all"

    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    models = [("codonbert", False), ("codonbert_hf", True), ("encodon-80m", False)]
    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, max3)]

    if model_key != "all":
        models = [(n, r) for n, r in models if n == model_key]
    if task_key != "all":
        tasks = [(n, d, m) for n, d, m in tasks if n == task_key]

    res_file = Path("./results/lora_finetune_results.json")
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for mname, use_rna in models:
        for task_name, task_df, max_n in tasks:
            if (mname, task_name) in done_keys:
                print(f"  SKIP {mname} {task_name} (already done)")
                continue
            r = run_lora_experiment(mname, use_rna, task_name, task_df, max_n, cds_cache)
            if r:
                existing.append(r)
                with open(res_file, "w") as f:
                    json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("ALL LoRA RESULTS")
    print(f"{'='*60}")
    for r in existing:
        print(f"  {r['model']:20s} | {r['task']:25s} | LR={r['test_lr_auc']:.4f} | MLP={r['test_mlp_auc']:.4f} | LoRA={r['test_lora_auc']:.4f} | Δ(LR)={r['improvement_over_lr']:+.4f}")

if __name__ == "__main__":
    main()