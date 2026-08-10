"""LoRA fine-tuning for extended models: EnCodon-620M, CodonTransformer, CaLM"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import src.models.xformers_compat
from src.eval.evaluation_utils import _fix_token_type_ids

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/lora_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16

MODELS = {
    "encodon-620m": {
        "hf_id": "goodarzilab/encodon-620M",
        "use_mlm": True, "rna": False,
    },
    "codontransformer": {
        "hf_id": "adibvafa/CodonTransformer",
        "use_mlm": True, "rna": False, "is_codontransformer": True,
    },
    "calm": {
        "hf_id": "oxpig/CaLM",
        "use_mlm": False, "rna": True, "is_calm": True, "use_pip_package": True,
    },
}

LORA_R = 8
LORA_ALPHA = 16
LORA_EPOCHS = 30
LORA_LR = 2e-4
LORA_WD = 0.01


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original = original_layer
        d_out, d_in = original_layer.weight.shape
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(model, r=8, alpha=16):
    n_injected = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in ['query', 'value', 'q_proj', 'v_proj']):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for n in parent_name.split('.'):
                if n:
                    parent = getattr(parent, n)
            lora_layer = LoRALayer(module, r=r, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            n_injected += 1
    return n_injected


def freeze_non_lora(model):
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def run_lora(model_name, model_info, sequences, labels, device="cuda:0"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

    hf_id = model_info["hf_id"]
    is_calm = model_info.get("is_calm", False)
    is_ct = model_info.get("is_codontransformer", False)
    rna = model_info.get("rna", False)
    use_pip = model_info.get("use_pip_package", False)

    print(f"  Loading {model_name}...")

    if is_calm and use_pip:
        from calm import CaLM
        calm_model = CaLM()
        tokenizer = calm_model.tokenizer
        model = calm_model.model.to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        try:
            model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True).to(device)
        except Exception:
            model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device)

    n_injected = inject_lora(model, r=LORA_R, alpha=LORA_ALPHA)
    print(f"  Injected LoRA into {n_injected} layers")
    model = model.to(device)
    freeze_non_lora(model)

    classifier = nn.Linear(model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768, 1).to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        range(len(sequences)), labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_seqs = [sequences[i] for i in X_train]
    test_seqs = [sequences[i] for i in X_test]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(classifier.parameters()),
        lr=LORA_LR, weight_decay=LORA_WD
    )
    criterion = nn.BCEWithLogitsLoss()

    print(f"  Training LoRA for {LORA_EPOCHS} epochs...")
    batch_size = 16

    for epoch in range(LORA_EPOCHS):
        model.train()
        classifier.train()
        perm = np.random.permutation(len(train_seqs))
        total_loss = 0

        for i in range(0, len(train_seqs), batch_size):
            idx = perm[i:i+batch_size]
            batch = [train_seqs[j] for j in idx]
            batch_labels = torch.tensor(y_train[idx], dtype=torch.float32).to(device)

            if is_ct:
                from Bio.Seq import Seq
                aa_map = {'F':'F','L':'L','I':'I','M':'M','V':'V','S':'S','P':'P','T':'T','A':'A','Y':'Y','H':'H','Q':'Q','N':'N','K':'K','D':'D','E':'E','C':'C','W':'W','R':'R','G':'G'}
                processed = []
                for seq in batch:
                    codons = seq.split()
                    aa_codons = []
                    for c in codons:
                        if len(c) == 3:
                            try:
                                aa = str(Seq(c).translate())
                            except:
                                aa = 'X'
                            aa_codons.append(f"{aa.lower()}_{c.lower()}")
                        else:
                            aa_codons.append(c)
                    processed.append(" ".join(aa_codons))
                batch = processed
            elif rna:
                batch = [s.replace('T', 'U').replace('t', 'u') for s in batch]

            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens = _fix_token_type_ids(tokens)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            out = model(**tokens, output_hidden_states=True)
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                emb = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            else:
                emb = out.logits

            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * attn_mask).sum(1) / attn_mask.sum(1)
            logits = classifier(pooled).squeeze(-1)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{LORA_EPOCHS}, Loss={total_loss/(len(train_seqs)/batch_size):.4f}")

    model.eval()
    classifier.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(test_seqs), batch_size):
            batch = test_seqs[i:i+batch_size]

            if is_ct:
                from Bio.Seq import Seq
                aa_map = {'F':'F','L':'L','I':'I','M':'M','V':'V','S':'S','P':'P','T':'T','A':'A','Y':'Y','H':'H','Q':'Q','N':'N','K':'K','D':'D','E':'E','C':'C','W':'W','R':'R','G':'G'}
                processed = []
                for seq in batch:
                    codons = seq.split()
                    aa_codons = []
                    for c in codons:
                        if len(c) == 3:
                            try:
                                aa = str(Seq(c).translate())
                            except:
                                aa = 'X'
                            aa_codons.append(f"{aa.lower()}_{c.lower()}")
                        else:
                            aa_codons.append(c)
                    processed.append(" ".join(aa_codons))
                batch = processed
            elif rna:
                batch = [s.replace('T', 'U').replace('t', 'u') for s in batch]

            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            tokens = _fix_token_type_ids(tokens)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            out = model(**tokens, output_hidden_states=True)
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                emb = out.hidden_states[-1]
            elif hasattr(out, 'last_hidden_state'):
                emb = out.last_hidden_state
            else:
                emb = out.logits

            attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * attn_mask).sum(1) / attn_mask.sum(1)
            logits = classifier(pooled).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())

    auc = roc_auc_score(y_test, all_preds)
    print(f"  LoRA Test AUC = {auc:.4f}")

    del model, classifier
    torch.cuda.empty_cache()
    return auc


def main():
    device = "cuda:0"

    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))

    print("Loading ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()

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

    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()

    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))

    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()

    tasks = [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, min(int((task3_variants["label"]==1).sum())*2, len(task3_variants))),
    ]

    all_results = []

    for mname, minfo in MODELS.items():
        for task_name, task_df, max_samples in tasks:
            print(f"\n{'='*60}")
            print(f"LoRA: {mname} on {task_name}")
            print(f"{'='*60}")

            df = task_df.copy()
            if len(df) > max_samples:
                n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
                patho = df[df["label"] == 1].sample(n=n_per_class, random_state=42)
                benign = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
                df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

            sequences = []
            labels = []
            for _, row in df.iterrows():
                tx = row["tx_id"]
                cpos = int(row["cpos"])
                cds_seq = cds_cache.get(tx, "")
                if not cds_seq:
                    continue
                codon_seq = build_codon_context(cds_seq, cpos)
                if codon_seq is None:
                    continue
                sequences.append(codon_seq)
                labels.append(row["label"])

            labels = np.array(labels, dtype=int)
            print(f"  {len(sequences)} samples, P={int(labels.sum())} B={int(len(labels)-labels.sum())}")

            try:
                auc = run_lora(mname, minfo, sequences, labels, device=device)
                all_results.append({
                    "model": mname, "task": task_name,
                    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                    "lora_epochs": LORA_EPOCHS,
                    "test_auc": round(float(auc), 4),
                    "n_samples": len(sequences),
                    "success": True
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": mname, "task": task_name, "success": False, "error": str(e)
                })

    out_file = OUT_DIR / "lora_extended_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("LoRA extended results:")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['model']:20s} | {r['task']:25s} | LoRA AUC={r['test_auc']:.4f}")
        else:
            print(f"  {r.get('model','?'):20s} | {r.get('task','?'):25s} | FAILED")


if __name__ == "__main__":
    main()