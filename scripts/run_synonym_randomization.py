import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, random, time, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from collections import defaultdict

DEVICE = "cuda:0"; DATA_DIR = Path("./data"); OUT_DIR = Path("./results/supplementary"); OUT_DIR.mkdir(parents=True, exist_ok=True); CODON_CONTEXT = 16

CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TABLE.items():
    if aa != '*':
        AA_TO_CODONS[aa].append(codon)

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def dna_to_rna(seq):
    return seq.replace("T", "U")

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds):
        return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def randomize_synonymous_codons(codon_str, variant_codon_idx=None, seed=42):
    rng = random.Random(seed)
    codons = codon_str.split()
    for i in range(len(codons)):
        if i == variant_codon_idx:
            continue
        c = codons[i]
        if c in CODON_TABLE and CODON_TABLE[c] != '*':
            synonyms = [x for x in AA_TO_CODONS[CODON_TABLE[c]] if x != c]
            if synonyms:
                codons[i] = rng.choice(synonyms)
    return " ".join(codons)

def load_data():
    cds = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()
    pk, bk = ["Pathogenic", "Likely pathogenic"], ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    v = snv[snv["label"] >= 0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len() == 1) & (v["AlternateAlleleVCF"].str.len() == 1)]
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"] = p.apply(lambda x: x[0]); v["cpos"] = p.apply(lambda x: x[1])
    h = v["tx_id"].notna() & v["cpos"].notna() & v["tx_id"].isin(set(cds.keys()))
    var = v[h].copy()
    var["is_syn"] = var["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    return cds, var

def balance(df, max_n):
    if max_n and len(df) > max_n:
        n = min(max_n // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
        return pd.concat([df[df["label"] == 1].sample(n, random_state=42),
                          df[df["label"] == 0].sample(n, random_state=42)]).sample(frac=1, random_state=42)
    return df

def extract_emb_for_model(mname, use_rna, seqs, device=DEVICE):
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import extract_embeddings
    model, tokenizer, meta = CodonModelLoader.load(mname, device=device)
    if model is None:
        print(f"  FAILED to load {mname}")
        return None
    if use_rna:
        seqs = [dna_to_rna(s) for s in seqs]
    emb = extract_embeddings(model, tokenizer, seqs, device=device, batch_size=16, show_progress=True)
    CodonModelLoader.release(model, device)
    return emb

def run_eval(emb, labs, label=""):
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labs, test_size=0.2, random_state=42, stratify=labs)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(X_tr, y_tr)
    lr_test = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    lr_cv = cross_val_score(LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"), emb, labs,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1).fit(X_tr, y_tr)
    mlp_test = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
    r = {
        "label": label, "n": int(len(labs)),
        "cv_lr_mean": round(float(lr_cv.mean()), 4), "cv_lr_std": round(float(lr_cv.std()), 4),
        "test_lr": round(float(lr_test), 4), "test_mlp": round(float(mlp_test), 4),
    }
    print(f"  {label}: CV-LR={r['cv_lr_mean']:.4f}±{r['cv_lr_std']:.4f} | Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")
    return r

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "codonbert"
    task_key = sys.argv[2] if len(sys.argv) > 2 else "synonymous"

    print(f"=== Synonymous Codon Randomization Ablation ===")
    print(f"Model: {model_key}, Task: {task_key}")

    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum()); nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    models = [("codonbert", False), ("codonbert_hf", True), ("encodon-80m", False)]
    tasks = [("missense", t2, 5000), ("synonymous", t3, max3)]

    use_rna_map = {m[0]: m[1] for m in models}

    results = []
    res_file = OUT_DIR / "synonym_randomization_results.json"

    for mname, use_rna in models:
        if model_key != "all" and mname != model_key:
            continue
        for task_name, task_df, max_n in tasks:
            if task_key != "all" and task_name != task_key:
                continue

            print(f"\n{'='*60}\n{mname} | {task_name}\n{'='*60}")

            df = balance(task_df.copy(), max_n)

            seqs_orig, seqs_rand, labs_list = [], [], []
            for _, row in df.iterrows():
                s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
                if s is None:
                    continue
                seqs_orig.append(s)
                s_rand = randomize_synonymous_codons(s, variant_codon_idx=None, seed=42)
                seqs_rand.append(s_rand)
                labs_list.append(row["label"])

            labs = np.array(labs_list, dtype=int)
            print(f"  {task_name}: {len(seqs_orig)} seqs")

            emb_orig_path = OUT_DIR / f"{mname}_{task_name}_emb.npy"
            lab_path = OUT_DIR / f"{mname}_{task_name}_labels.npy"

            if emb_orig_path.exists() and lab_path.exists():
                print(f"  Loading cached original embeddings from {emb_orig_path}")
                emb_orig = np.load(emb_orig_path)
                saved_labs = np.load(lab_path)
                assert len(saved_labs) == len(labs), f"Label mismatch: {len(saved_labs)} vs {len(labs)}"
            else:
                print(f"  Extracting ORIGINAL embeddings for {mname} {task_name}...")
                emb_orig = extract_emb_for_model(mname, use_rna, seqs_orig)
                if emb_orig is None:
                    continue
                np.save(emb_orig_path, emb_orig)
                np.save(lab_path, labs)

            emb_rand_path = OUT_DIR / f"{mname}_{task_name}_rand_emb.npy"
            if emb_rand_path.exists():
                print(f"  Loading cached randomized embeddings from {emb_rand_path}")
                emb_rand = np.load(emb_rand_path)
            else:
                print(f"  Extracting RANDOMIZED embeddings for {mname} {task_name}...")
                emb_rand = extract_emb_for_model(mname, use_rna, seqs_rand)
                if emb_rand is None:
                    continue
                np.save(emb_rand_path, emb_rand)

            print(f"\n  --- Original sequences ---")
            r_orig = run_eval(emb_orig, labs, label=f"{mname}_{task_name}_original")
            print(f"\n  --- Randomized synonymous codons ---")
            r_rand = run_eval(emb_rand, labs, label=f"{mname}_{task_name}_randomized")

            delta_lr = r_orig["test_lr"] - r_rand["test_lr"]
            delta_mlp = r_orig["test_mlp"] - r_rand["test_mlp"]

            entry = {
                "model": mname, "task": task_name,
                "original_cv_lr": r_orig["cv_lr_mean"], "original_cv_lr_std": r_orig["cv_lr_std"],
                "original_test_lr": r_orig["test_lr"], "original_test_mlp": r_orig["test_mlp"],
                "randomized_cv_lr": r_rand["cv_lr_mean"], "randomized_cv_lr_std": r_rand["cv_lr_std"],
                "randomized_test_lr": r_rand["test_lr"], "randomized_test_mlp": r_rand["test_mlp"],
                "delta_lr": round(delta_lr, 4), "delta_mlp": round(delta_mlp, 4),
                "n": int(len(labs)),
            }
            results.append(entry)
            print(f"\n  >>> DELTA: LR={delta_lr:+.4f}, MLP={delta_mlp:+.4f}")

            with open(res_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}\nALL RESULTS\n{'='*60}")
    for r in results:
        print(f"  {r['model']:15s} | {r['task']:15s} | Orig-LR={r['original_test_lr']:.4f} Rand-LR={r['randomized_test_lr']:.4f} ΔLR={r['delta_lr']:+.4f} | Orig-MLP={r['original_test_mlp']:.4f} Rand-MLP={r['randomized_test_mlp']:.4f} ΔMLP={r['delta_mlp']:+.4f}")

if __name__ == "__main__":
    main()