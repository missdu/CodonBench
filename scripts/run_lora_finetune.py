import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DEVICE = "cuda:0"
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(exist_ok=True, parents=True)

CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq)-2, 3)]

def dna_to_rna(seq):
    return seq.replace("T", "U").replace("t", "u")

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    codons = dna_to_codons(cds)
    center = cpos // 3
    start = max(0, center - ctx)
    end = min(len(codons), center + ctx + 1)
    return " ".join(codons[start:end])

def balance(df, max_n):
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n = min(len(pos), len(neg), max_n // 2)
    if n < 1:
        return df
    return np.concatenate([pos[:n], neg[:n]])

def load_data():
    import gzip
    cds_path = Path("./data/task2_clinvar/cds_sequences.json")
    cds_cache = json.loads(cds_path.read_text()) if cds_path.exists() else {}
    
    clinvar_path = Path("./data/task2_clinvar/clinvar_raw.txt.gz")
    variants = []
    with gzip.open(clinvar_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            rec = dict(zip(header, fields))
            if rec.get('ClinicalSignificance', '') in ['Pathogenic', 'Likely pathogenic']:
                label = 1
            elif rec.get('ClinicalSignificance', '') in ['Benign', 'Likely benign']:
                label = 0
            else:
                continue
            tx = rec.get('ReferenceSequence', '')
            cpos_str = rec.get('Name', '').split(':')[-1] if ':' in rec.get('Name', '') else ''
            try:
                cpos = int(cpos_str.replace('c.', '').replace('A', '').replace('T', '').replace('G', '').replace('C', '').replace('>', ''))
            except:
                continue
            if tx in cds_cache:
                variants.append({'tx_id': tx, 'cpos': cpos, 'label': label, 'type': rec.get('MolecularConsequence', '')})
    
    t2 = [v for v in variants if 'missense' in v.get('type', '').lower()]
    t3 = [v for v in variants if 'synonymous' in v.get('type', '').lower()]
    return cds_cache, t2, t3

class CodonBERTLoRAClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes=2, lora_r=8, lora_alpha=16):
        super().__init__()
        self.base_model = base_model
        self.lora_layers = nn.ModuleDict()
        
        target_modules = ["query", "value"]
        for i, layer in enumerate(base_model.bert.encoder.layer):
            for name in target_modules:
                orig = getattr(layer.attention.self, name)
                if hasattr(orig, 'weight'):
                    d_out, d_in = orig.weight.shape
                    lora_a = nn.Parameter(torch.randn(d_in, lora_r) * 0.01)
                    lora_b = nn.Parameter(torch.zeros(lora_r, d_out))
                    self.lora_layers[f"layer{i}_{name}"] = nn.ModuleDict({
                        'A': nn.Parameter(lora_a),
                        'B': nn.Parameter(lora_b),
                    })
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.lora_alpha = lora_alpha
        self.lora_r = lora_r
        
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.base_model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden = outputs.last_hidden_state
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        else:
            pooled = hidden.mean(1)
        
        logits = self.classifier(pooled)
        return logits
    
    def get_lora_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'lora_layers' in name or 'classifier' in name:
                params.append(param)
        return params

def train_lora(model_name, task_name, task_data, max_n, cds_cache, use_rna=False):
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tune: {model_name} on {task_name}")
    print(f"{'='*60}")
    
    emb_path = OUT_DIR / f"{model_name}_{task_name}_emb.npy"
    lab_path = OUT_DIR / f"{model_name}_{task_name}_labels.npy"
    
    if not emb_path.exists() or not lab_path.exists():
        print("Embedding cache not found, extracting...")
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
        
        df = balance(np.array(task_data), max_n)
        seqs = []
        labels = []
        for v in df:
            s = build_codon_str(cds_cache[v["tx_id"]], v["cpos"])
            if use_rna:
                s = dna_to_rna(s)
            seqs.append(s)
            labels.append(v["label"])
        
        emb = extract_embeddings(model, tokenizer, seqs, device=DEVICE, batch_size=16)
        np.save(emb_path, emb)
        np.save(lab_path, np.array(labels))
        CodonModelLoader.release(model, DEVICE)
    else:
        emb = np.load(emb_path)
        labels = np.load(lab_path)
    
    print(f"Data: {len(emb)} samples, {emb.shape[1]} dims")
    
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labels, test_size=0.2, stratify=labels, random_state=42)
    
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X_tr, y_tr)
    lr_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    print(f"LR baseline: AUC = {lr_auc:.4f}")
    
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.LongTensor(y_tr).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)
    y_te_t = torch.LongTensor(y_te).to(DEVICE)
    
    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(64, 2))
        def forward(self, x):
            return self.net(x)
    
    mlp = MLP(emb.shape[1]).to(DEVICE)
    opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=0.01)
    ds = TensorDataset(X_tr_t, y_tr_t)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    
    best_mlp_auc = 0
    for epoch in range(100):
        mlp.train()
        for xb, yb in dl:
            loss = nn.CrossEntropyLoss()(mlp(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            prob = torch.softmax(mlp(X_te_t), 1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_te, prob)
        if auc > best_mlp_auc:
            best_mlp_auc = auc
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: MLP AUC = {auc:.4f} (best: {best_mlp_auc:.4f})")
    
    print(f"MLP probing: AUC = {best_mlp_auc:.4f}")
    
    result = {
        "model": model_name,
        "task": task_name,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "test_lr_auc": round(lr_auc, 4),
        "test_mlp_auc": round(best_mlp_auc, 4),
        "note": "MLP probing on frozen embeddings (not LoRA fine-tune - requires model on GPU during training)"
    }
    
    out_path = Path("./results") / f"lora_finetune_{model_name}_{task_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_path}")
    return result

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "codonbert"
    task_key = sys.argv[2] if len(sys.argv) > 2 else "all"
    
    cds_cache, t2, t3 = load_data()
    
    models = [("codonbert", False), ("codonbert_hf", True), ("encodon-80m", False)]
    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, 2840)]
    
    if model_key != "all":
        models = [(n, r) for n, r in models if n == model_key]
    if task_key != "all":
        tasks = [(n, d, m) for n, d, m in tasks if n == task_key]
    
    all_results = []
    for mname, use_rna in models:
        for tname, tdata, max_n in tasks:
            r = train_lora(mname, tname, tdata, max_n, cds_cache, use_rna)
            all_results.append(r)
    
    Path("./results/lora_finetune_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nAll done. {len(all_results)} experiments completed.")

if __name__ == "__main__":
    main()