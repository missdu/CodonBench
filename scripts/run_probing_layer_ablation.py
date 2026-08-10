import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, time, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/ablation_studies")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
def dna_to_rna(seq): return seq.replace("T", "U")
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_cds_codon_str(cds_seq, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds_seq) // 3, ci + ctx + 1)
    codons = dna_to_codons(cds_seq[s*3:e*3])
    return " ".join(codons) if codons else None

def load_data():
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
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
    return cds_cache, variants

def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n, random_state=42), df[df["label"]==0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df

def main():
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import extract_embeddings
    
    cds_cache, variants = load_data()
    task3 = variants[variants["is_syn"]].copy()
    n_p = int((task3["label"]==1).sum())
    n_b = int((task3["label"]==0).sum())
    max3 = min(n_p*2, n_p+n_b)
    
    # Probing ablation
    for model_name, use_rna in [("codonbert_hf", True), ("codonbert", False)]:
        print(f"\n{'='*60}\nProbing ablation: {model_name} on task3_synonymous\n{'='*60}")
        config = CodonModelLoader.get_config(model_name)
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
        if model is None: continue
        
        df = balance(task3.copy(), max3)
        seqs, labs = [], []
        for _, row in df.iterrows():
            s = build_cds_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if s is None: continue
            if use_rna: s = dna_to_rna(s)
            seqs.append(s); labs.append(row["label"])
        labs = np.array(labs, dtype=int)
        
        emb = extract_embeddings(model, tokenizer, seqs, device=DEVICE, batch_size=16, show_progress=True)
        CodonModelLoader.release(model, DEVICE)
        
        probes = {"LR": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
                  "LR-C0.01": LogisticRegression(max_iter=2000, C=0.01, solver="lbfgs"),
                  "LR-C100": LogisticRegression(max_iter=2000, C=100, solver="lbfgs"),
                  "KNN-5": KNeighborsClassifier(n_neighbors=5),
                  "KNN-21": KNeighborsClassifier(n_neighbors=21),
                  "MLP": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        res = {}
        for name, clf in probes.items():
            aucs = cross_val_score(clf, emb, labs, cv=cv, scoring="roc_auc")
            res[name] = {"mean": round(float(aucs.mean()),4), "std": round(float(aucs.std()),4)}
            print(f"  {name}: {aucs.mean():.4f}+/-{aucs.std():.4f}")
        
        with open(OUT_DIR / f"probing_ablation_{model_name}_task3.json", "w") as f:
            json.dump(res, f, indent=2)
    
    # Layer ablation
    for model_name, use_rna in [("codonbert_hf", True)]:
        print(f"\n{'='*60}\nLayer ablation: {model_name} on task3_synonymous\n{'='*60}")
        config = CodonModelLoader.get_config(model_name)
        model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
        if model is None: continue
        
        df = balance(task3.copy(), max3)
        seqs, labs = [], []
        for _, row in df.iterrows():
            s = build_cds_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if s is None: continue
            if use_rna: s = dna_to_rna(s)
            seqs.append(s); labs.append(row["label"])
        labs = np.array(labs, dtype=int)
        
        n_layers = model.config.num_hidden_layers
        layers = sorted(set([1, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-1, n_layers]))
        print(f"  {n_layers} layers, testing: {layers}")
        
        layer_res = {}
        for layer in layers:
            all_embs = []
            for i in range(0, len(seqs), 16):
                batch = seqs[i:i+16]
                inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True)
                all_embs.append(out.hidden_states[layer][:, 0, :].cpu().numpy())
            emb = np.vstack(all_embs)
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = cross_val_score(clf, emb, labs, cv=cv, scoring="roc_auc")
            layer_res[str(layer)] = {"mean": round(float(aucs.mean()),4), "std": round(float(aucs.std()),4)}
            print(f"  Layer {layer}/{n_layers}: {aucs.mean():.4f}+/-{aucs.std():.4f}")
        
        CodonModelLoader.release(model, DEVICE)
        with open(OUT_DIR / f"layer_ablation_{model_name}_task3.json", "w") as f:
            json.dump({"n_layers": n_layers, "results": layer_res}, f, indent=2)

if __name__ == "__main__":
    main()