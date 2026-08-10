"""Step 4: ESM-1b baseline evaluation (add second pLM)"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/esm1b_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODON_TABLE = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
               'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
               'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
               'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
               'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
               'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
               'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
               'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}

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
    prot = "".join(CODON_TABLE.get(c, 'X') for c in codons).replace('*', '')
    return prot[:1022] if len(prot) >= 5 else None

def main():
    print("Loading data...")
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
    task2_df = variants[~variants["is_syn"]].copy()
    task3_df = variants[variants["is_syn"]].copy()

    import esm
    print("Loading ESM-1b...")
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    print(f"ESM-1b loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    results = {}
    for task_name, task_df, max_n in [("task2_missense", task2_df, 5000), ("task3_synonymous", task3_df, 2840)]:
        print(f"\n=== ESM-1b on {task_name} ===")
        df = task_df.copy()
        n_pc = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
        df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                        df[df["label"]==0].sample(n_pc, random_state=42)])

        prot_seqs, labels = [], []
        for _, row in df.iterrows():
            ps = build_protein_sequence(cds_cache[row["tx_id"]], row["cpos"])
            if ps: prot_seqs.append(ps); labels.append(row["label"])
        labels = np.array(labels)
        print(f"  {len(prot_seqs)} samples ({labels.sum()} pos, {(1-labels).sum()} neg)")

        print("  Extracting embeddings...")
        all_embeds = []
        batch_size = 4
        for i in range(0, len(prot_seqs), batch_size):
            batch_data = [(f"seq_{j}", prot_seqs[j]) for j in range(i, min(i+batch_size, len(prot_seqs)))]
            try:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(DEVICE)
                with torch.no_grad():
                    out = model(batch_tokens, repr_layers=[33])
                reps = out["representations"][33][:, 1:-1].mean(dim=1).cpu().numpy()
                all_embeds.append(reps)
            except Exception as e:
                print(f"  Error at batch {i}: {e}")
                for j in range(i, min(i+batch_size, len(prot_seqs))):
                    try:
                        single = [(f"seq_{j}", prot_seqs[j])]
                        _, _, toks = batch_converter(single)
                        toks = toks.to(DEVICE)
                        with torch.no_grad():
                            out = model(toks, repr_layers=[33])
                        rep = out["representations"][33][:, 1:-1].mean(dim=1).cpu().numpy()
                        all_embeds.append(rep)
                    except:
                        pass

        X = np.concatenate(all_embeds, axis=0)
        print(f"  Embeddings: {X.shape}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
            lr.fit(X[train_idx], labels[train_idx])
            pred = lr.predict_proba(X[test_idx])[:, 1]
            auc = roc_auc_score(labels[test_idx], pred)
            fold_aucs.append(auc)
            print(f"  Fold {fold}: AUC = {auc:.4f}")

        results[task_name] = {
            "model": "ESM-1b-650M",
            "task": task_name,
            "model_type": "pLM",
            "params_M": 650,
            "n_samples": len(X),
            "ROC-AUC_mean": float(np.mean(fold_aucs)),
            "ROC-AUC_std": float(np.std(fold_aucs)),
            "ROC-AUC_folds": [float(x) for x in fold_aucs],
        }
        print(f"  Mean AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    out_path = OUT_DIR / "esm1b_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()