import sys
sys.path.insert(0, ".")
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import json
import re
import time
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from scipy import stats

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/ablation_studies")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16
CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {c: i for i, c in enumerate(CODONS)}

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def dna_to_rna(seq):
    return seq.replace("T", "U")

def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None

def build_cds_codon_str(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)

def build_protein_sequence(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    subseq = cds_seq[start_codon*3:end_codon*3]
    codons = dna_to_codons(subseq)
    if not codons:
        return None
    codon_table = {
        'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
        'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
        'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
        'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
        'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
        'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
        'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
        'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
    }
    protein = "".join(codon_table.get(c, 'X') for c in codons)
    protein = protein.replace('*', '')
    return protein if len(protein) >= 5 else None

def build_onehot_pos_features(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    max_len = 33
    emb = np.zeros(max_len * 64, dtype=np.float32)
    for i, c in enumerate(codons[:max_len]):
        if c in CODON_TO_IDX:
            emb[i * 64 + CODON_TO_IDX[c]] = 1.0
    return emb

def load_clinvar_data(cds_cache):
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
    return variants

def balance_sample(df, max_samples, random_state=42):
    if max_samples and len(df) > max_samples:
        n_per_class = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        patho = df[df["label"] == 1].sample(n=n_per_class, random_state=random_state)
        benign = df[df["label"] == 0].sample(n=n_per_class, random_state=random_state)
        return pd.concat([patho, benign]).sample(frac=1, random_state=random_state)
    return df

# ============================================================
# P0: ESM-2 (Protein Language Model) baseline
# ============================================================
def evaluate_esm2(task_name, variants_df, cds_cache, max_samples=5000):
    print("\n" + "="*60)
    print(f"P0: Evaluating ESM-2 (pLM) on {task_name}")
    print("="*60)
    
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"  Loaded ESM-2 650M, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    df = balance_sample(variants_df.copy(), max_samples)
    
    sequences = []
    labels = []
    skipped = 0
    for _, row in df.iterrows():
        prot = build_protein_sequence(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if prot is None or len(prot) < 5:
            skipped += 1
            continue
        if len(prot) > 1022:
            prot = prot[:1022]
        sequences.append(prot)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(sequences)} protein sequences (skipped {skipped})")
    
    if len(sequences) < 100:
        del model
        torch.cuda.empty_cache()
        return {"model": "ESM-2-650M", "task": task_name, "success": False, "error": "Too few"}
    
    all_embs = []
    batch_size = 8
    t0 = time.time()
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        data = [(f"seq_{j}", s) for j, s in enumerate(batch_seqs)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(DEVICE)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reprs = results["representations"][33]
        cls_embs = token_reprs[:, 0, :].cpu().numpy()
        all_embs.append(cls_embs)
        
        if (i // batch_size) % 20 == 0:
            print(f"    Batch {i//batch_size+1}/{(len(sequences)-1)//batch_size+1}")
    
    embeddings = np.vstack(all_embs)
    emb_time = time.time() - t0
    print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    
    result = run_probing(embeddings, labels, "ESM-2-650M", task_name, emb_time)
    result["model_type"] = "pLM"
    result["params_M"] = 650
    result["input_type"] = "protein_sequence"
    return result

# ============================================================
# P0: onehot_pos with independent test set
# ============================================================
def evaluate_onehot_pos_independent(task_name, variants_df, cds_cache, max_samples=5000):
    print("\n" + "="*60)
    print(f"P0: onehot_pos with independent test set on {task_name}")
    print("="*60)
    
    df = balance_sample(variants_df.copy(), max_samples)
    
    features = []
    labels = []
    skipped = 0
    for _, row in df.iterrows():
        feat = build_onehot_pos_features(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if feat is None:
            skipped += 1
            continue
        features.append(feat)
        labels.append(row["label"])
    
    features = np.array(features)
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(features)} samples (skipped {skipped}), dim={features.shape[1]}")
    
    if len(features) < 100:
        return {"model": "onehot_pos", "task": task_name, "success": False}
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    for name, clf in [
        ("LR", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ("KNN-5", KNeighborsClassifier(n_neighbors=5)),
        ("KNN-21", KNeighborsClassifier(n_neighbors=21)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ]:
        t0 = time.time()
        clf.fit(X_train, y_train)
        from sklearn.metrics import roc_auc_score
        y_pred = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred)
        train_time = time.time() - t0
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_aucs = cross_val_score(clf, features, labels, cv=cv, scoring="roc_auc")
        
        results[name] = {
            "test_auc": round(float(test_auc), 4),
            "cv_auc_mean": round(float(cv_aucs.mean()), 4),
            "cv_auc_std": round(float(cv_aucs.std()), 4),
            "train_time_s": round(train_time, 2),
        }
        print(f"  {name}: test_auc={test_auc:.4f}, cv_auc={cv_aucs.mean():.4f}+/-{cv_aucs.std():.4f}")
    
    result = {
        "model": "onehot_pos", "task": task_name, "success": True,
        "model_type": "Traditional", "n_samples": len(features),
        "n_train": len(X_train), "n_test": len(X_test),
        "feat_dim": features.shape[1],
        "probing_results": results,
    }
    
    with open(OUT_DIR / f"onehot_pos_independent_{task_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

# ============================================================
# P1: Probing method ablation (using cLM embeddings)
# ============================================================
def evaluate_probing_ablation(model_name, task_name, variants_df, cds_cache, use_rna=False, max_samples=5000):
    print("\n" + "="*60)
    print(f"P1: Probing ablation for {model_name} on {task_name}")
    print("="*60)
    
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import extract_embeddings
    
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    if model is None:
        return {"model": model_name, "task": task_name, "success": False}
    
    df = balance_sample(variants_df.copy(), max_samples)
    
    sequences = []
    labels = []
    for _, row in df.iterrows():
        seq_str = build_cds_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if seq_str is None:
            continue
        if use_rna:
            seq_str = dna_to_rna(seq_str)
        sequences.append(seq_str)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(sequences)} sequences")
    
    embeddings = extract_embeddings(model, tokenizer, sequences, device=DEVICE, batch_size=16, show_progress=True)
    CodonModelLoader.release(model, DEVICE)
    
    probing_methods = {
        "LR": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        "LR-C0.01": LogisticRegression(max_iter=2000, C=0.01, solver="lbfgs"),
        "LR-C100": LogisticRegression(max_iter=2000, C=100, solver="lbfgs"),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "KNN-21": KNeighborsClassifier(n_neighbors=21),
        "SVM-RBF": SVC(kernel="rbf", probability=True, C=1.0),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, clf in probing_methods.items():
        t0 = time.time()
        aucs = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")
        cv_time = time.time() - t0
        results[name] = {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std()), 4),
            "time_s": round(cv_time, 1),
        }
        print(f"  {name}: AUC={aucs.mean():.4f}+/-{aucs.std():.4f} ({cv_time:.1f}s)")
    
    result = {
        "model": model_name, "task": task_name, "success": True,
        "n_samples": len(sequences), "probing_results": results,
    }
    with open(OUT_DIR / f"probing_ablation_{model_name}_{task_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

# ============================================================
# P1: Embedding layer ablation
# ============================================================
def evaluate_layer_ablation(model_name, task_name, variants_df, cds_cache, use_rna=False, max_samples=5000):
    print("\n" + "="*60)
    print(f"P1: Layer ablation for {model_name} on {task_name}")
    print("="*60)
    
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    
    config = CodonModelLoader.get_config(model_name)
    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    if model is None:
        return {"model": model_name, "task": task_name, "success": False}
    
    df = balance_sample(variants_df.copy(), max_samples)
    
    sequences = []
    labels = []
    for _, row in df.iterrows():
        seq_str = build_cds_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if seq_str is None:
            continue
        if use_rna:
            seq_str = dna_to_rna(seq_str)
        sequences.append(seq_str)
        labels.append(row["label"])
    
    labels = np.array(labels, dtype=int)
    print(f"  Built {len(sequences)} sequences")
    
    n_layers = model.config.num_hidden_layers
    layers_to_test = [1, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-1, n_layers]
    layers_to_test = sorted(set(max(1, l) for l in layers_to_test))
    print(f"  Model has {n_layers} layers, testing: {layers_to_test}")
    
    layer_results = {}
    for layer in layers_to_test:
        all_embs = []
        batch_size = 16
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer]
                embs = hidden[:, 0, :].cpu().numpy()
            all_embs.append(embs)
        
        embeddings = np.vstack(all_embs)
        
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")
        
        layer_results[str(layer)] = {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std()), 4),
            "emb_dim": embeddings.shape[1],
        }
        print(f"  Layer {layer}/{n_layers}: AUC={aucs.mean():.4f}+/-{aucs.std():.4f}")
    
    CodonModelLoader.release(model, DEVICE)
    
    result = {
        "model": model_name, "task": task_name, "success": True,
        "n_layers": n_layers, "layer_results": layer_results,
    }
    with open(OUT_DIR / f"layer_ablation_{model_name}_{task_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result

# ============================================================
# Helper: run probing with multiple methods
# ============================================================
def run_probing(embeddings, labels, model_name, task_name, emb_time_s=0):
    probing_methods = {
        "LR": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}
    
    for name, clf in probing_methods.items():
        t0 = time.time()
        aucs = cross_val_score(clf, embeddings, labels, cv=cv, scoring="roc_auc")
        cv_time = time.time() - t0
        all_results[name] = {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std": round(float(aucs.std()), 4),
            "auc_folds": [round(float(a), 4) for a in aucs],
        }
        print(f"  {name}: AUC={aucs.mean():.4f}+/-{aucs.std():.4f}")
    
    best_name = max(all_results, key=lambda k: all_results[k]["auc_mean"])
    best = all_results[best_name]
    
    result = {
        "model": model_name, "task": task_name,
        "success": True, "n_samples": len(embeddings),
        "n_pathogenic": int(labels.sum()), "n_benign": int(len(labels) - labels.sum()),
        "best_probing": best_name,
        "ROC-AUC_mean": best["auc_mean"],
        "ROC-AUC_std": best["auc_std"],
        "ROC-AUC_folds": best["auc_folds"],
        "all_probing": all_results,
        "emb_time_s": round(emb_time_s, 1),
    }
    
    with open(OUT_DIR / f"{model_name}_{task_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


# ============================================================
# Main
# ============================================================
def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    print(f"  {len(cds_cache)} transcripts")
    
    variants = load_clinvar_data(cds_cache)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()
    print(f"  Task2 (missense): {len(task2_variants):,}")
    print(f"  Task3 (synonymous): {len(task3_variants):,}")
    
    n_task3_patho = int((task3_variants["label"]==1).sum())
    n_task3_benign = int((task3_variants["label"]==0).sum())
    task3_max = min(n_task3_patho * 2, n_task3_patho + n_task3_benign)
    
    all_results = []
    
    # ---- P0: ESM-2 (pLM baseline) ----
    for task_name, task_df, max_samples in [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]:
        try:
            r = evaluate_esm2(task_name, task_df, cds_cache, max_samples)
            all_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({"model": "ESM-2-650M", "task": task_name, "success": False, "error": str(e)})
    
    # ---- P0: onehot_pos independent test set ----
    for task_name, task_df, max_samples in [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]:
        try:
            r = evaluate_onehot_pos_independent(task_name, task_df, cds_cache, max_samples)
            all_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            all_results.append({"model": "onehot_pos_indep", "task": task_name, "success": False, "error": str(e)})
    
    # ---- P1: Probing ablation for best cLM (CodonBERT) ----
    for task_name, task_df, max_samples in [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]:
        try:
            r = evaluate_probing_ablation("codonbert", task_name, task_df, cds_cache, use_rna=False, max_samples=max_samples)
            all_results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
    
    # ---- P1: Layer ablation for CodonBERT ----
    try:
        r = evaluate_layer_ablation("codonbert", "task3_synonymous", task3_variants, cds_cache, use_rna=False, max_samples=task3_max)
        all_results.append(r)
    except Exception as e:
        import traceback; traceback.print_exc()
    
    # Save summary
    with open(OUT_DIR / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDIES COMPLETE")
    print("="*60)
    for r in all_results:
        if r.get("success"):
            if "ROC-AUC_mean" in r:
                print(f"  {r['model']:20s} | {r['task']:30s} | AUC={r['ROC-AUC_mean']:.4f}+/-{r['ROC-AUC_std']:.4f} | best_probing={r.get('best_probing','N/A')}")
            elif "probing_results" in r:
                best_p = max(r["probing_results"], key=lambda k: r["probing_results"][k].get("auc_mean", 0) if isinstance(r["probing_results"][k], dict) else r["probing_results"][k].get("test_auc", 0))
                print(f"  {r['model']:20s} | {r['task']:30s} | best_probing={best_p}")
            elif "layer_results" in r:
                best_l = max(r["layer_results"], key=lambda k: r["layer_results"][k]["auc_mean"])
                print(f"  {r['model']:20s} | {r['task']:30s} | best_layer={best_l}, AUC={r['layer_results'][best_l]['auc_mean']:.4f}")
        else:
            print(f"  {r.get('model','?'):20s} | {r.get('task','?'):30s} | FAILED")

if __name__ == "__main__":
    main()