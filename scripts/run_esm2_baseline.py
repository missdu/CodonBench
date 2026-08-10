import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/ablation_studies")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_protein_sequence(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq): return None
    codon_idx = (cpos - 1) // 3
    start = max(0, codon_idx - context_codons)
    end = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start*3:end*3])
    if not codons: return None
    ct = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
          'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
          'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
          'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
          'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
          'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
          'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
          'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
    prot = "".join(ct.get(c, 'X') for c in codons).replace('*', '')
    return prot[:1022] if len(prot) >= 5 else None

def main():
    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
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
    valid = snv[snv["label"] >= 0].copy()
    valid = valid[(valid["ReferenceAlleleVCF"].str.len()==1) & (valid["AlternateAlleleVCF"].str.len()==1)]
    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    
    task2 = variants[~variants["is_syn"]].copy()
    task3 = variants[variants["is_syn"]].copy()
    print(f"Task2: {len(task2):,}, Task3: {len(task3):,}")
    
    import esm
    print("Loading ESM-2...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"ESM-2 loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    results = {}
    for task_name, task_df, max_n in [("task2_missense", task2, 5000), ("task3_synonymous", task3, 2840)]:
        print(f"\n=== ESM-2 on {task_name} ===")
        df = task_df.copy()
        if len(df) > max_n:
            n_pc = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
            df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                           df[df["label"]==0].sample(n_pc, random_state=42)]).sample(frac=1, random_state=42)
        
        seqs, labs = [], []
        for _, row in df.iterrows():
            p = build_protein_sequence(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if p: seqs.append(p); labs.append(row["label"])
        labs = np.array(labs, dtype=int)
        print(f"  {len(seqs)} protein sequences")
        
        if len(seqs) < 100: continue
        
        all_embs = []
        bs = 4
        t0 = time.time()
        for i in range(0, len(seqs), bs):
            data = [(f"s{j}", s) for j, s in enumerate(seqs[i:i+bs])]
            _, _, tokens = batch_converter(data)
            with torch.no_grad():
                out = model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            all_embs.append(out["representations"][33][:, 0, :].cpu().numpy())
            if (i//bs) % 50 == 0:
                print(f"  Batch {i//bs+1}/{(len(seqs)-1)//bs+1}")
        emb = np.vstack(all_embs)
        emb_time = time.time() - t0
        print(f"  Embeddings: {emb.shape}, {emb_time:.1f}s")
        
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(clf, emb, labs, cv=cv, scoring="roc_auc")
        
        r = {
            "model": "ESM-2-650M", "task": task_name, "model_type": "pLM",
            "params_M": 650, "input_type": "protein_sequence",
            "success": True, "n_samples": len(seqs),
            "ROC-AUC_mean": round(float(aucs.mean()), 4),
            "ROC-AUC_std": round(float(aucs.std()), 4),
            "ROC-AUC_folds": [round(float(a), 4) for a in aucs],
            "emb_time_s": round(emb_time, 1),
        }
        results[task_name] = r
        print(f"  AUC={aucs.mean():.4f}+/-{aucs.std():.4f}")
        
        with open(OUT_DIR / f"esm2_{task_name}.json", "w") as f:
            json.dump(r, f, indent=2)
    
    del model
    torch.cuda.empty_cache()
    
    print("\n=== ESM-2 RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: AUC={v['ROC-AUC_mean']:.4f}+/-{v['ROC-AUC_std']:.4f}")

if __name__ == "__main__":
    main()