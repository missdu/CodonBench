import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DEVICE = "cuda:0"
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CODON_CONTEXT = 16

CASES = [
    ("PAH:c.1197A>T", "NM_000277.3", 1197, 1, "PKU"),
    ("KCNQ1:c.1032G>A", "NM_000218.3", 1032, 1, "Long QT"),
    ("BRCA2:c.9117G>A", "NM_000059.4", 9117, 1, "HBOC"),
    ("MSH2:c.2634G>A", "NM_000251.3", 2634, 1, "Lynch"),
    ("F8:c.5217C>T", "NM_000132.4", 5217, 1, "Hemophilia A"),
    ("GCK:c.1285A>C", "NM_000162.5", 1285, 0, "MODY (benign)"),
    ("MLH1:c.1959G>T", "NM_000249.4", 1959, 0, "Lynch (benign)"),
    ("MSH2:c.1666T>C", "NM_000251.3", 1666, 0, "Lynch (benign)"),
]

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

def extract_attention_and_embedding(model, tokenizer, sequence, device=DEVICE):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if "token_type_ids" in inputs and hasattr(model, "bert"):
        if inputs["token_type_ids"].shape[1] != inputs["input_ids"].shape[1]:
            diff = inputs["input_ids"].shape[1] - inputs["token_type_ids"].shape[1]
            inputs["token_type_ids"] = torch.cat([
                inputs["token_type_ids"],
                torch.zeros(inputs["token_type_ids"].shape[0], diff, dtype=torch.long, device=device)
            ], dim=1)
    
    forward_params = set(model.forward.__code__.co_varnames) if hasattr(model, 'forward') else set()
    filtered = {k: v for k, v in inputs.items() if k in forward_params or k in {'input_ids', 'attention_mask', 'token_type_ids'}}
    
    with torch.no_grad():
        outputs = model(**filtered, output_attentions=True, output_hidden_states=True)
    
    result = {"tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())}
    
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        n_layers = len(outputs.attentions)
        attn_np = np.stack([a[0].cpu().numpy() for a in outputs.attentions])
        result["attention"] = attn_np
        result["n_layers"] = n_layers
        result["n_heads"] = attn_np.shape[1]
    else:
        result["attention"] = None
        result["n_layers"] = 0
        result["n_heads"] = 0
    
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        last_hidden = outputs.hidden_states[-1][0].cpu().numpy()
        mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))[0].cpu().numpy()
        pooled = (last_hidden * mask[:, None]).sum(0) / mask.sum()
        result["embedding"] = pooled
    else:
        result["embedding"] = None
    
    return result

def compute_gradient_attribution(model, tokenizer, sequence, device=DEVICE):
    model.eval()
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if "token_type_ids" in inputs:
        if inputs["token_type_ids"].shape[1] != inputs["input_ids"].shape[1]:
            diff = inputs["input_ids"].shape[1] - inputs["token_type_ids"].shape[1]
            inputs["token_type_ids"] = torch.cat([
                inputs["token_type_ids"],
                torch.zeros(inputs["token_type_ids"].shape[0], diff, dtype=torch.long, device=device)
            ], dim=1)
    
    forward_params = set(model.forward.__code__.co_varnames) if hasattr(model, 'forward') else set()
    filtered = {k: v for k, v in inputs.items() if k in forward_params or k in {'input_ids', 'attention_mask', 'token_type_ids'}}
    
    embed_layer = model.get_input_embeddings()
    input_ids = filtered["input_ids"]
    embeds = embed_layer(input_ids).detach().requires_grad_(True)
    
    outputs = model(inputs_embeds=embeds, attention_mask=filtered.get("attention_mask"), 
                    token_type_ids=filtered.get("token_type_ids"))
    
    if hasattr(outputs, 'logits') and outputs.logits is not None:
        score = outputs.logits[0].sum()
    elif hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
        score = outputs.prediction_logits[0].sum()
    elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
        mask = filtered.get("attention_mask", torch.ones_like(input_ids)).float().unsqueeze(-1)
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        score = pooled.sum()
    else:
        return None
    
    score.backward()
    grad = embeds.grad[0].cpu().numpy()
    attr = np.abs(grad * embeds[0].detach().cpu().numpy())
    token_attr = attr.sum(axis=-1)
    
    return token_attr

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "codonbert"
    
    cds_path = Path("./data/task2_clinvar/cds_sequences.json")
    cds_cache = json.loads(cds_path.read_text()) if cds_path.exists() else {}
    
    models_to_run = [("codonbert", False), ("codonbert_hf", True), ("encodon-80m", False)]
    if model_key != "all":
        models_to_run = [(n, r) for n, r in models_to_run if n == model_key]
    
    all_results = {}
    
    for mname, use_rna in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {mname}")
        print(f"{'='*60}")
        
        model, tokenizer, meta = CodonModelLoader.load(mname, device=DEVICE)
        
        model_results = {"cases": {}, "tsne_data": {}}
        
        case_embeddings = []
        case_labels = []
        case_names = []
        
        for case_name, tx_id, cpos, true_label, desc in CASES:
            if tx_id not in cds_cache:
                print(f"  Skip {case_name}: no CDS for {tx_id}")
                continue
            
            cds = cds_cache[tx_id]
            codon_str = build_codon_str(cds, cpos)
            if use_rna:
                codon_str = dna_to_rna(codon_str)
            
            print(f"  Processing {case_name}...")
            
            try:
                result = extract_attention_and_embedding(model, tokenizer, codon_str, DEVICE)
                
                case_result = {
                    "case": case_name,
                    "tx_id": tx_id,
                    "cpos": cpos,
                    "true_label": true_label,
                    "description": desc,
                    "n_tokens": len(result["tokens"]),
                    "n_layers": result["n_layers"],
                    "n_heads": result["n_heads"],
                }
                
                if result["attention"] is not None:
                    attn_path = OUT_DIR / f"{mname}_{case_name.replace(':','_')}_attention.npy"
                    np.save(attn_path, result["attention"])
                    case_result["attention_path"] = str(attn_path)
                    
                    avg_attn = result["attention"].mean(axis=(0, 1))
                    center_idx = len(result["tokens"]) // 2
                    local_attn = avg_attn[center_idx, max(0, center_idx-5):center_idx+6]
                    case_result["center_local_attention"] = local_attn.tolist()
                
                if result["embedding"] is not None:
                    case_embeddings.append(result["embedding"])
                    case_labels.append(true_label)
                    case_names.append(case_name)
                
                try:
                    token_attr = compute_gradient_attribution(model, tokenizer, codon_str, DEVICE)
                    if token_attr is not None:
                        attr_path = OUT_DIR / f"{mname}_{case_name.replace(':','_')}_gradient.npy"
                        np.save(attr_path, token_attr)
                        case_result["gradient_path"] = str(attr_path)
                        center_idx = len(token_attr) // 2
                        case_result["center_gradient"] = float(token_attr[center_idx])
                except Exception as e:
                    print(f"    Gradient attribution failed: {e}")
                
                model_results["cases"][case_name] = case_result
                
            except Exception as e:
                print(f"    Failed: {e}")
        
        if case_embeddings:
            all_emb = np.array(case_embeddings)
            
            if len(all_emb) >= 5:
                tsne = TSNE(n_components=2, perplexity=min(30, len(all_emb)-1), random_state=42)
                coords = tsne.fit_transform(all_emb)
                tsne_path = OUT_DIR / f"{mname}_case_study_tsne.npy"
                np.save(tsne_path, coords)
                model_results["tsne_coords"] = coords.tolist()
                model_results["tsne_labels"] = case_labels
                model_results["tsne_names"] = case_names
        
        all_results[mname] = model_results
        CodonModelLoader.release(model, DEVICE)
    
    emb_tasks = [("task2_missense", 5000), ("task3_synonymous", 2840)]
    for mname, use_rna in models_to_run:
        for tname, max_n in emb_tasks:
            emb_path = OUT_DIR / f"{mname}_{tname}_emb.npy"
            lab_path = OUT_DIR / f"{mname}_{tname}_labels.npy"
            
            if emb_path.exists() and lab_path.exists():
                print(f"\nGenerating t-SNE for {mname} {tname}...")
                emb = np.load(emb_path)
                labs = np.load(lab_path)
                
                n = min(2000, len(emb))
                idx = np.random.choice(len(emb), n, replace=False)
                emb_sub = emb[idx]
                labs_sub = labs[idx]
                
                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                coords = tsne.fit_transform(emb_sub)
                
                tsne_path = OUT_DIR / f"{mname}_{tname}_tsne.npy"
                np.save(tsne_path, coords)
                np.save(OUT_DIR / f"{mname}_{tname}_tsne_labels.npy", labs_sub)
                print(f"  Saved t-SNE: {coords.shape}")
    
    out_path = Path("./results/attention_analysis_results.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAll done. Saved to {out_path}")

if __name__ == "__main__":
    main()