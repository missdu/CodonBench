"""ESM-2 + ESM-1b MLP probing experiment (P0-1).

Protocol matches cLM MLP probing exactly:
  - 80/20 train/test split (stratified, random_state=42)
  - MLP: hidden_layer_sizes=(128,64), max_iter=300, early_stopping=True
  - LR: 5-fold CV (stratified, random_state=42) + independent test set
  - Balanced sampling: MisPath n=5000, SynPath n=2840

Critical test: if ESM-2 MLP on SynPath improves significantly (like cLMs),
the channel-specificity claim collapses. If it stays ~0.680, claim is supported.
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

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
    if len(df) > max_n:
        n = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n, random_state=42),
                          df[df["label"]==0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df

def extract_esm_embeddings(model_name, model, batch_converter, prot_seqs, device=DEVICE):
    """Extract CLS token embeddings from ESM model."""
    all_embs = []
    bs = 4
    t0 = time.time()
    for i in range(0, len(prot_seqs), bs):
        data = [(f"s{j}", prot_seqs[j]) for j in range(i, min(i+bs, len(prot_seqs)))]
        try:
            _, _, tokens = batch_converter(data)
            with torch.no_grad():
                out = model(tokens.to(device), repr_layers=[33], return_contacts=False)
            all_embs.append(out["representations"][33][:, 0, :].cpu().numpy())
        except Exception as e:
            print(f"  Batch error at {i}: {e}, falling back to single...")
            for j in range(i, min(i+bs, len(prot_seqs))):
                try:
                    single = [(f"s{j}", prot_seqs[j])]
                    _, _, toks = batch_converter(single)
                    with torch.no_grad():
                        out = model(toks.to(device), repr_layers=[33], return_contacts=False)
                    all_embs.append(out["representations"][33][:, 0, :].cpu().numpy())
                except:
                    pass
        if (i//bs) % 50 == 0:
            print(f"  Batch {i//bs+1}/{(len(prot_seqs)-1)//bs+1}")
    emb = np.vstack(all_embs)
    elapsed = time.time() - t0
    print(f"  Embeddings: {emb.shape}, {elapsed:.1f}s")
    return emb

def run_model(model_name, task_name, task_df, max_n, cds_cache):
    """Run ESM model on one task: extract embeddings + LR + MLP probing."""
    import esm

    emb_path = OUT_DIR / f"{model_name}_{task_name}_emb.npy"
    lab_path = OUT_DIR / f"{model_name}_{task_name}_labels.npy"

    if emb_path.exists() and lab_path.exists():
        print(f"  Loading cached embeddings from {emb_path}")
        emb = np.load(emb_path)
        labs = np.load(lab_path)
    else:
        print(f"  Loading {model_name}...")
        if model_name == "ESM-2-650M":
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_name == "ESM-1b-650M":
            model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        model = model.to(DEVICE).eval()
        batch_converter = alphabet.get_batch_converter()
        print(f"  {model_name} loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

        df = balance(task_df.copy(), max_n)
        prot_seqs, labs_list = [], []
        for _, row in df.iterrows():
            ps = build_protein_sequence(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
            if ps:
                prot_seqs.append(ps)
                labs_list.append(row["label"])
        labs = np.array(labs_list, dtype=int)
        print(f"  {task_name}: {len(prot_seqs)} protein sequences")

        emb = extract_esm_embeddings(model_name, model, batch_converter, prot_seqs)
        np.save(emb_path, emb)
        np.save(lab_path, labs)
        del model
        torch.cuda.empty_cache()

    print(f"  Embeddings shape: {emb.shape}, Labels: {len(labs)}")

    X_tr, X_te, y_tr, y_te = train_test_split(emb, labs, test_size=0.2, random_state=42, stratify=labs)

    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
    lr_test = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    lr_cv = cross_val_score(
        LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
        emb, labs, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc"
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
        early_stopping=True, validation_fraction=0.1
    ).fit(X_tr, y_tr)
    mlp_test = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])

    r = {
        "model": model_name,
        "task": task_name,
        "model_type": "pLM",
        "n": int(len(labs)),
        "cv_lr_mean": round(float(lr_cv.mean()), 4),
        "cv_lr_std": round(float(lr_cv.std()), 4),
        "test_lr": round(float(lr_test), 4),
        "test_mlp": round(float(mlp_test), 4),
        "lr_to_mlp_gain": round(float(mlp_test - lr_test), 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    print(f"  CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
          f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
          f"LR→MLP gain={r['lr_to_mlp_gain']:+.4f}")
    return r

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "all"
    task_key = sys.argv[2] if len(sys.argv) > 2 else "all"

    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    models = ["ESM-2-650M", "ESM-1b-650M"]
    tasks = [
        ("task2_missense", t2, 5000),
        ("task3_synonymous", t3, max3),
    ]

    res_file = OUT_DIR / "esm_mlp_probing_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for mname in models:
        if model_key != "all" and mname != model_key:
            continue
        for task_name, task_df, max_n in tasks:
            if task_key != "all" and task_name != task_key:
                continue
            if (mname, task_name) in done_keys:
                print(f"  SKIP {mname} {task_name} (already done)")
                continue
            print(f"\n{'='*60}\n{mname} | {task_name}\n{'='*60}")
            r = run_model(mname, task_name, task_df, max_n, cds_cache)
            if r:
                existing.append(r)
                with open(res_file, "w") as f:
                    json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nALL ESM MLP PROBING RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:15s} | {r['task']:25s} | "
              f"CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | "
              f"Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f} | "
              f"LR→MLP={r['lr_to_mlp_gain']:+.4f}")

    print("\n=== KEY COMPARISON: cLM vs pLM LR→MLP gain ===")
    print("  cLMs on SynPath: LR→MLP gain = +10 to +23 pp (from existing results)")
    print("  pLMs on SynPath: check above — if gain < 5 pp, channel-specificity SUPPORTED")
    print("  pLMs on MisPath: expected gain ≈ 0 pp (pLMs already near ceiling on AA channel)")

if __name__ == "__main__":
    main()