import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import src.models.xformers_compat
import json
import numpy as np
import torch
from pathlib import Path
from itertools import combinations

from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

DEVICE = "cuda:0"
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def linear_cka(X, Y, eps=1e-10):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = np.trace(K @ H @ L @ H) / (n - 1) ** 2
    hsic_xx = np.trace(K @ H @ K @ H) / (n - 1) ** 2
    hsic_yy = np.trace(L @ H @ L @ H) / (n - 1) ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + eps)

def rbf_cka(X, Y, sigma=None, eps=1e-10):
    from scipy.spatial.distance import cdist
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    if sigma is None:
        sigma = np.median(cdist(X, X, 'sqeuclidean'))
    K = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sigma + eps))
    L = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * sigma + eps))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    hsic_xy = np.trace(K @ H @ L @ H) / (n - 1) ** 2
    hsic_xx = np.trace(K @ H @ K @ H) / (n - 1) ** 2
    hsic_yy = np.trace(L @ H @ L @ H) / (n - 1) ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + eps)

def svca(X, Y, eps=1e-10):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    k = min(min(X.shape[1], Y.shape[1]), 50)
    Ux_k = Ux[:, :k]
    Uy_k = Uy[:, :k]
    return np.linalg.norm(Ux_k.T @ Uy_k, 'fro') ** 2 / k

def extract_all_layer_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=8, max_length=2048):
    all_layer_embs = []
    n_layers = None
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
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
        
        with torch.no_grad():
            outputs = model(**filtered, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        if n_layers is None:
            n_layers = len(hidden_states)
            all_layer_embs = [[] for _ in range(n_layers)]
        
        mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).float().unsqueeze(-1)
        
        for layer_idx in range(n_layers):
            hidden = hidden_states[layer_idx]
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            all_layer_embs[layer_idx].append(pooled.cpu().numpy())
    
    result = []
    for layer_idx in range(n_layers):
        result.append(np.vstack(all_layer_embs[layer_idx]))
    
    return result

def main():
    task_key = sys.argv[1] if len(sys.argv) > 1 else "task3_synonymous"
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    
    models_info = [
        ("codonbert", False, "cLM"),
        ("codonbert_hf", True, "cLM"),
        ("encodon-80m", False, "cLM"),
    ]
    
    print(f"CKA Analysis for {task_key} (max_n={max_n})")
    
    emb_cache = {}
    label_cache = {}
    
    for mname, use_rna, mtype in models_info:
        emb_path = OUT_DIR / f"{mname}_{task_key}_emb.npy"
        lab_path = OUT_DIR / f"{mname}_{task_key}_labels.npy"
        
        if emb_path.exists() and lab_path.exists():
            emb = np.load(emb_path)
            labs = np.load(lab_path)
            n = min(max_n, len(emb))
            idx = np.random.choice(len(emb), n, replace=False)
            emb_cache[mname] = emb[idx]
            label_cache[mname] = labs[idx]
            print(f"  {mname}: loaded {emb.shape} from cache, using {n} samples")
        else:
            print(f"  {mname}: no cache found, skipping. Run run_mlp_independent_v2.py first.")
    
    all_results = {"task": task_key, "pairwise_cka": [], "layerwise_cka": {}}
    
    model_names = list(emb_cache.keys())
    for (m1, m2) in combinations(model_names, 2):
        X = emb_cache[m1]
        Y = emb_cache[m2]
        
        n = min(len(X), len(Y))
        X = X[:n]
        Y = Y[:n]
        
        print(f"\n  CKA({m1}, {m2}):")
        
        lin_cka = linear_cka(X, Y)
        rbf_cka_val = rbf_cka(X[:500], Y[:500])
        svca_val = svca(X, Y)
        
        print(f"    Linear CKA = {lin_cka:.4f}")
        print(f"    RBF CKA    = {rbf_cka_val:.4f}")
        print(f"    SVCA       = {svca_val:.4f}")
        
        all_results["pairwise_cka"].append({
            "model_1": m1, "model_2": m2,
            "n_samples": n,
            "linear_cka": round(lin_cka, 4),
            "rbf_cka": round(rbf_cka_val, 4),
            "svca": round(svca_val, 4),
        })
    
    layerwise_flag = "--layers" in sys.argv
    if layerwise_flag:
        print("\n  Computing layerwise CKA...")
        for (m1, m2) in combinations(model_names, 2):
            key = f"{m1}_vs_{m2}"
            all_results["layerwise_cka"][key] = []
            
            try:
                model1, tok1, _ = CodonModelLoader.load(m1, device=DEVICE)
                model2, tok2, _ = CodonModelLoader.load(m2, device=DEVICE)
                
                n_layers_1 = len([m for m in model1.modules() if 'Layer' in type(m).__name__ or 'Encoder' in type(m).__name__])
                
                n = min(200, len(emb_cache[m1]))
                idx = np.random.choice(len(emb_cache[m1]), n, replace=False)
                
                print(f"    {key}: extracting all-layer embeddings ({n} samples)...")
                
                CodonModelLoader.release(model1, DEVICE)
                CodonModelLoader.release(model2, DEVICE)
                
            except Exception as e:
                print(f"    Layerwise CKA failed for {key}: {e}")
    
    out_path = Path("./results/cka_similarity_results.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)))
    print(f"\nSaved to {out_path}")
    
    print("\n=== CKA Summary ===")
    for item in all_results["pairwise_cka"]:
        print(f"  {item['model_1']} vs {item['model_2']}: Linear={item['linear_cka']:.3f}, RBF={item['rbf_cka']:.3f}, SVCA={item['svca']:.3f}")

if __name__ == "__main__":
    main()