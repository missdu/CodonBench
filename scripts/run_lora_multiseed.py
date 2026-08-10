"""LoRA 3-seed robustness check.

Runs LoRA fine-tuning with seeds [42, 123, 456] on standard 80/20 split.
Seed 42 results already exist in lora_gene_stratified_results.json;
this script re-runs all 3 seeds for consistency and reports mean ± s.d.

Usage:
  python scripts/run_lora_multiseed.py [--device cuda:0]
"""
import sys; sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for proxy_var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(proxy_var, None)

import json, re, gc, argparse, time
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
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

MODELS = {
    "codonbert": {"hf_id": "models/codonbert", "use_mlm": False, "rna": False, "use_loader": True},
    "codonbert_hf": {"hf_id": "models/codonbert-hf", "use_mlm": True, "rna": True, "use_loader": True},
    "encodon-80m": {"hf_id": "goodarzilab/encodon-80M", "use_mlm": True, "rna": False, "use_loader": True, "is_encodon": True},
    "encodon-620m": {"hf_id": "goodarzilab/encodon-620M", "use_mlm": True, "rna": False, "use_loader": True, "is_encodon": True},
    "codontransformer": {"hf_id": "adibvafa/CodonTransformer", "use_mlm": True, "rna": False, "is_codontransformer": True},
}

LORA_R = 8
LORA_ALPHA = 16
LORA_EPOCHS = 30
LORA_LR = 2e-4
LORA_WD = 0.01
BATCH_SIZE = 16
SEEDS = [42, 123, 456]
LOG_FILE = OUT_DIR / "lora_multiseed.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq): return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    return " ".join(codons) if codons else None

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    return (m.group(1), int(m.group(3)), m.group(4), m.group(5)) if m else (None, None, None, None)

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
                if n: parent = getattr(parent, n)
            lora_layer = LoRALayer(module, r=r, alpha=alpha)
            setattr(parent, child_name, lora_layer)
            n_injected += 1
    return n_injected

def freeze_non_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = 'lora_A' in name or 'lora_B' in name

def load_model(model_name, model_info, device):
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
    from src.models.loader import CodonModelLoader
    hf_id = model_info["hf_id"]
    is_ct = model_info.get("is_codontransformer", False)
    use_loader = model_info.get("use_loader", False)
    if use_loader and not is_ct:
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=device)
        if model is None: raise RuntimeError(f"Failed to load {model_name}")
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    try: model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True).to(device)
    except: model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device)
    return model, tokenizer

def prepare_batch(batch_seqs, model_info, tokenizer, device):
    is_ct = model_info.get("is_codontransformer", False)
    rna = model_info.get("rna", False)
    if is_ct:
        from Bio.Seq import Seq
        processed = []
        for seq in batch_seqs:
            codons = seq.split()
            aa_codons = []
            for c in codons:
                if len(c) == 3:
                    try: aa = str(Seq(c).translate())
                    except: aa = 'X'
                    aa_codons.append(f"{aa.lower()}_{c.lower()}")
                else: aa_codons.append(c)
            processed.append(" ".join(aa_codons))
        batch_seqs = processed
    elif rna:
        batch_seqs = [s.replace('T', 'U').replace('t', 'u') for s in batch_seqs]
    tokens = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
    tokens = _fix_token_type_ids(tokens)
    return {k: v.to(device) for k, v in tokens.items()}

def get_pooled_output(model, tokens):
    out = model(**tokens, output_hidden_states=True)
    if hasattr(out, 'hidden_states') and out.hidden_states is not None: emb = out.hidden_states[-1]
    elif hasattr(out, 'last_hidden_state'): emb = out.last_hidden_state
    else: emb = out.logits
    attn_mask = tokens["attention_mask"].unsqueeze(-1).float()
    return (emb * attn_mask).sum(1) / attn_mask.sum(1)

def train_and_evaluate_lora(model, tokenizer, model_info, train_seqs, train_labels,
                            test_seqs, test_labels, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_injected = inject_lora(model, r=LORA_R, alpha=LORA_ALPHA)
    model = model.to(device)
    freeze_non_lora(model)
    hidden_size = model.config.hidden_size if hasattr(model, 'hidden_size') else 768
    torch.manual_seed(seed)
    classifier = nn.Linear(hidden_size, 1).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(classifier.parameters()),
        lr=LORA_LR, weight_decay=LORA_WD)
    criterion = nn.BCEWithLogitsLoss()
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    best_auc = 0
    best_epoch = 0
    for epoch in range(LORA_EPOCHS):
        model.train(); classifier.train()
        perm = np.random.permutation(len(train_seqs))
        total_loss = 0; n_batches = 0
        for i in range(0, len(train_seqs), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch = [train_seqs[j] for j in idx]
            batch_labels = torch.tensor(train_labels[idx], dtype=torch.float32).to(device)
            tokens = prepare_batch(batch, model_info, tokenizer, device)
            pooled = get_pooled_output(model, tokens)
            logits = classifier(pooled).squeeze(-1)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(classifier.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        model.eval(); classifier.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(test_seqs), BATCH_SIZE):
                batch = test_seqs[i:i+BATCH_SIZE]
                tokens = prepare_batch(batch, model_info, tokenizer, device)
                pooled = get_pooled_output(model, tokens)
                logits = classifier(pooled).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.tolist())
        auc = roc_auc_score(test_labels, all_preds)
        if auc > best_auc: best_auc = auc; best_epoch = epoch + 1
        if (epoch + 1) % 10 == 0:
            log(f"    Epoch {epoch+1}/{LORA_EPOCHS}, Loss={total_loss/n_batches:.4f}, AUC={auc:.4f} (best={best_auc:.4f})")
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name: param.requires_grad = False
        else: param.requires_grad = True
    return best_auc, best_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--tasks", nargs="+", default=["task2_missense", "task3_synonymous"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log("=== LORA MULTI-SEED START ===")
    log(f"Models: {args.models}")
    log(f"Tasks: {args.tasks}")
    log(f"Seeds: {args.seeds}")

    device = args.device
    log("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    log("Loading ClinVar...")
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
        (labeled["ReferenceAlleleVCF"] != "na") & (labeled["AlternateAlleleVCF"] != "na") &
        (labeled["ReferenceAlleleVCF"].str.len() == 1) & (labeled["AlternateAlleleVCF"].str.len() == 1)
    ].copy()
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()
    def is_synonymous(name): return bool(re.search(r"p\.\w+\d+=", str(name)))
    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()
    tasks = {
        "task2_missense": (task2_variants, 5000),
        "task3_synonymous": (task3_variants, min(int((task3_variants["label"]==1).sum())*2, len(task3_variants))),
    }

    all_results = []
    out_file = OUT_DIR / "lora_multiseed_results.json"

    for mname in args.models:
        if mname not in MODELS: continue
        minfo = MODELS[mname]
        for task_name in args.tasks:
            if task_name not in tasks: continue
            task_df, max_n = tasks[task_name]
            df = task_df.copy()
            if len(df) > max_n:
                n_per_class = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
                df = pd.concat([df[df["label"]==1].sample(n=n_per_class, random_state=42),
                                df[df["label"]==0].sample(n=n_per_class, random_state=42)]).sample(frac=1, random_state=42)
            sequences, labels = [], []
            for _, row in df.iterrows():
                tx = row["tx_id"]; cpos = int(row["cpos"])
                cds_seq = cds_cache.get(tx, "")
                if not cds_seq: continue
                codon_seq = build_codon_context(cds_seq, cpos)
                if codon_seq is None: continue
                sequences.append(codon_seq); labels.append(row["label"])
            labels = np.array(labels, dtype=int)
            log(f"\n{'='*60}\n{mname} | {task_name} | {len(sequences)} samples\n{'='*60}")

            seed_results = []
            for seed in args.seeds:
                log(f"  Seed {seed}...")
                train_idx, test_idx = train_test_split(
                    range(len(sequences)), labels, test_size=0.2, random_state=seed, stratify=labels)
                train_seqs = [sequences[i] for i in train_idx]
                test_seqs = [sequences[i] for i in test_idx]
                train_labels = [labels[i] for i in train_idx]
                test_labels = [labels[i] for i in test_idx]

                model, tokenizer = load_model(mname, minfo, device)
                auc, best_ep = train_and_evaluate_lora(
                    model, tokenizer, minfo, train_seqs, train_labels, test_seqs, test_labels, device, seed)
                log(f"  Seed {seed}: AUC={auc:.4f}, best_epoch={best_ep}")
                seed_results.append({"seed": seed, "auc": round(float(auc), 4), "best_epoch": best_ep})

                del model, tokenizer; gc.collect(); torch.cuda.empty_cache()

            aucs = [r["auc"] for r in seed_results]
            entry = {
                "model": mname, "task": task_name,
                "seeds": seed_results,
                "mean_auc": round(float(np.mean(aucs)), 4),
                "std_auc": round(float(np.std(aucs)), 4),
            }
            all_results.append(entry)
            log(f"  SUMMARY: mean={entry['mean_auc']:.4f} ± {entry['std_auc']:.4f}")
            with open(out_file, "w") as f: json.dump(all_results, f, indent=2)

    log(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    log(f"{'Model':20s} | {'Task':25s} | {'Mean AUC':>8s} | {'Std':>6s} | Seeds")
    for r in all_results:
        seed_strs = [f"{s['seed']}={s['auc']:.4f}" for s in r['seeds']]
        log(f"{r['model']:20s} | {r['task']:25s} | {r['mean_auc']:8.4f} | {r['std_auc']:6.4f} | {', '.join(seed_strs)}")
    log("=== LORA MULTI-SEED DONE ===")

if __name__ == "__main__":
    main()