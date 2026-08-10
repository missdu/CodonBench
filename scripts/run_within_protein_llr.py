"""P1-2: Within-protein zero-shot LLR evaluation.

CodonTransformer succeeded at zero-shot on within-protein tasks
but our previous zero-shot evaluation used cross-protein LLR
which showed AUC≈0.5. This script tests within-protein LLR
to explain the discrepancy.

For each variant, we compute:
  LLR = log P(mutant_sequence) - log P(wildtype_sequence)
using the same gene's CDS as context (not a different protein).

Only codon-tokenized cLMs can do this (need per-token log probabilities).
"""
import sys, os, json, re, time
import torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

# Add project root to sys.path for src.* imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results")
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

def dna_to_rna_codons(seq):
    return [seq[i:i+3].replace("T","U") for i in range(0, len(seq) - len(seq) % 3, 3)]

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3)), m.group(4), m.group(5)) if m else (None, None, None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds) // 3, ci + ctx + 1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def build_mutant_codon_str(cds, cpos, ref_allele, alt_allele, ctx=CODON_CONTEXT):
    """Build mutant CDS context by substituting the variant nucleotide."""
    if cpos < 1 or cpos > len(cds): return None
    # Substitute at cpos (1-indexed)
    cds_list = list(cds)
    pos0 = cpos - 1
    if cds_list[pos0] != ref_allele:
        return None  # Reference mismatch
    cds_list[pos0] = alt_allele
    mutant_cds = "".join(cds_list)
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(mutant_cds) // 3, ci + ctx + 1)
    c = dna_to_codons(mutant_cds[s*3:e*3])
    return " ".join(c) if c else None

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
    valid["ref_allele"] = parsed.apply(lambda x: x[2])
    valid["alt_allele"] = parsed.apply(lambda x: x[3])
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

def compute_llr(model, tokenizer, wt_seq, mut_seq, device=DEVICE):
    """Compute log-likelihood ratio: log P(mutant) - log P(wildtype)."""
    from src.eval.evaluation_utils import _fix_token_type_ids

    def get_log_prob(seq):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs = _fix_token_type_ids(inputs)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return -outputs.loss.item() * inputs["input_ids"].shape[1]

    try:
        wt_lp = get_log_prob(wt_seq)
        mut_lp = get_log_prob(mut_seq)
        return mut_lp - wt_lp
    except:
        return None

def run_model(model_name, use_rna, task_name, task_df, max_n, cds_cache):
    import src.models.xformers_compat  # noqa: F401 — patches xformers for EnCodon
    from src.models.loader import CodonModelLoader

    print(f"\n{'='*60}\n{model_name} | {task_name} (within-protein LLR)\n{'='*60}")

    model, tokenizer, meta = CodonModelLoader.load(model_name, device=DEVICE)
    if model is None:
        print(f"  FAILED to load {model_name}")
        return None

    df = balance(task_df.copy(), max_n)
    llrs, labels = [], []
    n_ref_mismatch = 0

    for _, row in df.iterrows():
        cds = cds_cache.get(row["tx_id"], "")
        cpos = int(row["cpos"])
        ref = row["ref_allele"]
        alt = row["alt_allele"]

        wt_seq = build_codon_str(cds, cpos)
        mut_seq = build_mutant_codon_str(cds, cpos, ref, alt)

        if wt_seq is None or mut_seq is None:
            n_ref_mismatch += 1
            continue

        if use_rna:
            wt_seq = wt_seq.replace("T", "U")
            mut_seq = mut_seq.replace("T", "U")

        llr = compute_llr(model, tokenizer, wt_seq, mut_seq)
        if llr is not None:
            llrs.append(llr)
            labels.append(row["label"])

    CodonModelLoader.release(model, DEVICE)

    llrs = np.array(llrs)
    labels = np.array(labels)
    print(f"  Valid LLRs: {len(llrs)}, ref mismatches: {n_ref_mismatch}")

    if len(llrs) < 100 or len(set(labels)) < 2:
        print(f"  Too few samples or single class, skipping")
        return None

    # Higher LLR = mutant more likely = benign (label=0)
    # So we negate for pathogenic prediction
    auc = roc_auc_score(labels, -llrs)

    r = {
        "model": model_name,
        "task": task_name,
        "eval_type": "within_protein_llr",
        "n": int(len(llrs)),
        "n_ref_mismatch": n_ref_mismatch,
        "auc": round(float(auc), 4),
        "llr_mean_pathogenic": round(float(llrs[labels==1].mean()), 4),
        "llr_mean_benign": round(float(llrs[labels==0].mean()), 4),
    }
    print(f"  AUC={r['auc']:.4f} | LLR mean: patho={r['llr_mean_pathogenic']:.4f}, benign={r['llr_mean_benign']:.4f}")
    return r

def main():
    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"] == 1).sum())
    nb3 = int((t3["label"] == 0).sum())
    max3 = min(np3 * 2, np3 + nb3)

    models = [("codonbert", False), ("codonbert_hf", True), ("encodon-80m", False)]
    tasks = [("task2_missense", t2, 5000), ("task3_synonymous", t3, max3)]

    res_file = OUT_DIR / "within_protein_llr_results.json"
    existing = []
    if res_file.exists():
        existing = json.load(open(res_file))
    done_keys = {(r["model"], r["task"]) for r in existing}

    for mname, use_rna in models:
        for task_name, task_df, max_n in tasks:
            if (mname, task_name) in done_keys:
                print(f"  SKIP {mname} {task_name}")
                continue
            r = run_model(mname, use_rna, task_name, task_df, max_n, cds_cache)
            if r:
                existing.append(r)
                with open(res_file, "w") as f:
                    json.dump(existing, f, indent=2, default=str)

    print(f"\n{'='*60}\nWITHIN-PROTEIN LLR RESULTS\n{'='*60}")
    for r in existing:
        print(f"  {r['model']:20s} | {r['task']:25s} | AUC={r['auc']:.4f} | n={r['n']}")

if __name__ == "__main__":
    main()