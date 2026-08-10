"""Extract embeddings for from-scratch v4 model (original + randomized)."""
import os, sys, json, re, random, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from collections import defaultdict
from transformers import BertForMaskedLM

DEVICE = "cuda:0"
EXP_DIR = Path("./")
DATA_DIR = EXP_DIR / "data"
MODEL_DIR = EXP_DIR / "from_scratch_models"
OUT_DIR = EXP_DIR / "results" / "supplementary"
CODON_CONTEXT = 128

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
    if aa != '*': AA_TO_CODONS[aa].append(codon)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
VOCAB = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}

class CodonTokenizer:
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.vocab = VOCAB
    def __call__(self, text, max_length=None):
        ml = max_length or self.max_length
        codons = text.split() if " " in text else [text[i:i+3] for i in range(0, len(text)-2, 3)]
        codons = codons[:ml - 2]
        ids = [VOCAB["[CLS]"]] + [VOCAB.get(c, VOCAB["[UNK]"]) for c in codons] + [VOCAB["[SEP]"]]
        return {"input_ids": ids}

def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
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
            if synonyms: codons[i] = rng.choice(synonyms)
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

def extract_cls_embeddings(model_path, tokenizer, seqs, device=DEVICE, batch_size=16):
    print(f"Loading model from {model_path}...")
    model = BertForMaskedLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    print(f"Model loaded. Extracting embeddings for {len(seqs)} sequences...")

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            max_len = min(max(len(s.split()) for s in batch) + 2, tokenizer.max_length)
            all_ids = []
            for s in batch:
                enc = tokenizer(s, max_length=max_len)
                all_ids.append(enc["input_ids"])
            max_l = max(len(ids) for ids in all_ids)
            padded = [ids + [VOCAB["[PAD]"]] * (max_l - len(ids)) for ids in all_ids]
            input_ids = torch.tensor(padded, dtype=torch.long).to(device)
            attention_mask = (input_ids != VOCAB["[PAD]"]).long().to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)
            if (i // batch_size) % 10 == 0:
                print(f"  Batch {i//batch_size}/{len(seqs)//batch_size}")

    emb = np.concatenate(all_embs, axis=0)
    del model
    torch.cuda.empty_cache()
    print(f"Extracted embeddings: {emb.shape}")
    return emb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["v3b", "v4"])
    parser.add_argument("--task", required=True, choices=["synonymous", "missense"])
    args = parser.parse_args()

    model_map = {
        "v3b": MODEL_DIR / "codon-bert-ablation-v3b-ensembl",
        "v4": MODEL_DIR / "codon-bert-ablation-v4-scale110m",
    }
    model_path = model_map[args.model]
    tokenizer = CodonTokenizer(max_length=512)

    cds_cache, variants = load_data()
    if args.task == "synonymous":
        task_df = variants[variants["is_syn"]].copy()
        np3 = int((task_df["label"] == 1).sum()); nb3 = int((task_df["label"] == 0).sum())
        max_n = min(np3 * 2, np3 + nb3)
    else:
        task_df = variants[~variants["is_syn"]].copy()
        max_n = 5000

    df = balance(task_df.copy(), max_n)
    seqs_orig, seqs_rand, labs_list = [], [], []
    for _, row in df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"], ""), int(row["cpos"]))
        if s is None: continue
        seqs_orig.append(s)
        seqs_rand.append(randomize_synonymous_codons(s, seed=42))
        labs_list.append(row["label"])

    labs = np.array(labs_list, dtype=int)
    print(f"{args.model} {args.task}: {len(seqs_orig)} seqs")

    emb_orig_path = OUT_DIR / f"from_scratch_{args.model}_{args.task}_ctx128_emb.npy"
    emb_rand_path = OUT_DIR / f"from_scratch_{args.model}_{args.task}_ctx128_rand_emb.npy"
    lab_path = OUT_DIR / f"from_scratch_{args.model}_{args.task}_ctx128_labels.npy"

    if not emb_orig_path.exists():
        emb_orig = extract_cls_embeddings(model_path, tokenizer, seqs_orig)
        np.save(emb_orig_path, emb_orig)
        np.save(lab_path, labs)
    else:
        print(f"Cached: {emb_orig_path}")

    if not emb_rand_path.exists():
        emb_rand = extract_cls_embeddings(model_path, tokenizer, seqs_rand)
        np.save(emb_rand_path, emb_rand)
    else:
        print(f"Cached: {emb_rand_path}")

    print("Done!")

if __name__ == "__main__":
    main()