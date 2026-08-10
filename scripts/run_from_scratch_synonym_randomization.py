"""
Synonym randomization for from-scratch models (v3b, v4).
Tests whether small-scale models encode gene identity (randomization helps)
or genuine synonymous signal (randomization hurts).
"""
import os, sys, json, re, random, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from transformers import BertForMaskedLM, BertConfig

DEVICE = "cuda:0"
EXP_DIR = Path("./")
DATA_DIR = EXP_DIR / "data"
MODEL_DIR = EXP_DIR / "from_scratch_models"
OUT_DIR = EXP_DIR / "results" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

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

CODON_VOCAB = [c for c, aa in CODON_TABLE.items() if aa != '*']

AA_TO_CODONS = defaultdict(list)
for codon, aa in CODON_TABLE.items():
    if aa != '*':
        AA_TO_CODONS[aa].append(codon)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}


class CodonTokenizer:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.vocab = VOCAB
        self.id_to_token = {i: t for t, i in VOCAB.items()}

    def __call__(self, text, truncation=True, max_length=None, padding=False, return_tensors=None):
        ml = max_length or self.max_length
        codons = text.split() if " " in text else [text[i:i+3] for i in range(0, len(text)-2, 3)]
        if truncation:
            codons = codons[:ml - 2]
        ids = [VOCAB["[CLS]"]] + [VOCAB.get(c, VOCAB["[UNK]"]) for c in codons] + [VOCAB["[SEP]"]]
        attention_mask = [1] * len(ids)
        result = {"input_ids": ids, "attention_mask": attention_mask}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([ids])
            result["attention_mask"] = torch.tensor([attention_mask])
        return result

    @property
    def pad_token_id(self):
        return VOCAB["[PAD]"]


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


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


def randomize_synonymous_codons(codon_str, seed=42):
    rng = random.Random(seed)
    codons = codon_str.split()
    for i in range(len(codons)):
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


def extract_embeddings_from_scratch(model_dir, tokenizer, seqs, device=DEVICE, batch_size=32):
    model = BertForMaskedLM.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            max_len = min(max(len(s.split()) for s in batch) + 2, tokenizer.max_length)
            encoded = [tokenizer(s, max_length=max_len, return_tensors="pt") for s in batch]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [e["input_ids"][0] for e in encoded],
                batch_first=True, padding_value=VOCAB["[PAD]"]
            ).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [e["attention_mask"][0] for e in encoded],
                batch_first=True, padding_value=0
            ).to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)

    emb = np.concatenate(all_embs, axis=0)
    del model
    torch.cuda.empty_cache()
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


MODEL_MAP = {
    "v3b": MODEL_DIR / "codon-bert-ablation-v3b-ensembl",
    "v4": MODEL_DIR / "codon-bert-ablation-v4-scale110m",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["v3b", "v4", "all"])
    parser.add_argument("--task", default="all", choices=["synonymous", "missense", "all"])
    args = parser.parse_args()

    print("=== From-Scratch Synonym Randomization ===")
    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum()); nb3 = int((t3["label"] == 0).sum())
    max_syn = min(np3 * 2, np3 + nb3)

    tasks = [("synonymous", t3, max_syn), ("missense", t2, 5000)]
    models_to_run = ["v3b", "v4"] if args.model == "all" else [args.model]
    tasks_to_run = tasks if args.task == "all" else [(t, df, n) for t, df, n in tasks if t == args.task]

    tokenizer = CodonTokenizer(max_length=512)
    results = []
    res_file = OUT_DIR / "from_scratch_synonym_randomization_results.json"

    for mname in models_to_run:
        model_path = MODEL_MAP[mname]
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        for task_name, task_df, max_n in tasks_to_run:
            print(f"\n{'='*60}\n{mname} | {task_name}\n{'='*60}")
            df = balance(task_df.copy(), max_n)

            seqs_orig, seqs_rand, labs_list = [], [], []
            for _, row in df.iterrows():
                s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
                if s is None:
                    continue
                seqs_orig.append(s)
                s_rand = randomize_synonymous_codons(s, seed=42)
                seqs_rand.append(s_rand)
                labs_list.append(row["label"])

            labs = np.array(labs_list, dtype=int)
            print(f"  {task_name}: {len(seqs_orig)} seqs")

            emb_orig_path = OUT_DIR / f"from_scratch_{mname}_{task_name}_emb.npy"
            emb_rand_path = OUT_DIR / f"from_scratch_{mname}_{task_name}_rand_emb.npy"
            lab_path = OUT_DIR / f"from_scratch_{mname}_{task_name}_labels.npy"

            if emb_orig_path.exists():
                print(f"  Loading cached original embeddings")
                emb_orig = np.load(emb_orig_path)
            else:
                print(f"  Extracting ORIGINAL embeddings for {mname} {task_name}...")
                emb_orig = extract_embeddings_from_scratch(model_path, tokenizer, seqs_orig)
                np.save(emb_orig_path, emb_orig)
                np.save(lab_path, labs)

            if emb_rand_path.exists():
                print(f"  Loading cached randomized embeddings")
                emb_rand = np.load(emb_rand_path)
            else:
                print(f"  Extracting RANDOMIZED embeddings for {mname} {task_name}...")
                emb_rand = extract_embeddings_from_scratch(model_path, tokenizer, seqs_rand)
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
        print(f"  {r['model']:5s} | {r['task']:12s} | Orig-MLP={r['original_test_mlp']:.4f} Rand-MLP={r['randomized_test_mlp']:.4f} ΔMLP={r['delta_mlp']:+.4f}")


if __name__ == "__main__":
    main()