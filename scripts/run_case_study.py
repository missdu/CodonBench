import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:0"; DATA_DIR = Path("./data"); OUT_DIR = Path("./results/supplementary"); CODON_CONTEXT = 16

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
def dna_to_rna(seq): return seq.replace("T", "U")
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)
def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos<1 or cpos>len(cds): return None
    ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
    c=dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

cds_cache = json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
cols = ["Name","Type","ClinicalSignificance","ReviewStatus","GeneSymbol","ReferenceAlleleVCF","AlternateAlleleVCF"]
raw = pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",usecols=cols,low_memory=False)
snv = raw[raw["Type"]=="single nucleotide variant"].copy()
pk,bk=["Pathogenic","Likely pathogenic"],["Benign","Likely benign"]
def classify(cs):
    cs=str(cs)
    if any(k in cs for k in pk): return 1
    if any(k in cs for k in bk): return 0
    return -1
snv["label"]=snv["ClinicalSignificance"].apply(classify)
v=snv[snv["label"]>=0].copy()
v=v[(v["ReferenceAlleleVCF"].str.len()==1)&(v["AlternateAlleleVCF"].str.len()==1)]
p=v["Name"].apply(parse_hgvs)
v["tx_id"]=p.apply(lambda x:x[0]); v["cpos"]=p.apply(lambda x:x[1])
h=v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds_cache.keys()))
var=v[h].copy()
syn=var[var["Name"].apply(lambda n:bool(re.search(r"p\.\w+\d+=",str(n))))].copy()

cases = [
    ("PAH", "NM_000277.3", 1197, 1, "Phenylketonuria, reviewed by expert panel"),
    ("ATM", "NM_000051.4", 3576, 1, "Ataxia-telangiectasia, reviewed by expert panel"),
    ("KCNQ1", "NM_000218.3", 1032, 1, "Long QT syndrome, reviewed by expert panel"),
    ("LDLR", "NM_000527.5", 1216, 1, "Familial hypercholesterolemia, reviewed by expert panel"),
    ("CFTR", "NM_000492.4", 2988, 1, "Cystic fibrosis, reviewed by expert panel"),
    ("BRCA2", "NM_000059.4", 9117, 1, "Hereditary breast/ovarian cancer, reviewed by expert panel"),
    ("MLH1", "NM_000249.4", 1038, 1, "Lynch syndrome, reviewed by expert panel"),
    ("MSH2", "NM_000251.3", 2634, 1, "Lynch syndrome, reviewed by expert panel"),
    ("APC", "NM_000038.6", 1956, 1, "Familial adenomatous polyposis, reviewed by expert panel"),
    ("F8", "NM_000132.4", 5217, 1, "Hemophilia A, reviewed by expert panel"),
    ("ACADVL", "NM_000018.4", 1038, 0, "VLCAD deficiency - benign, reviewed by expert panel"),
    ("GCK", "NM_000162.5", 1285, 0, "Maturity-onset diabetes - benign, reviewed by expert panel"),
    ("MLH1_ben", "NM_000249.4", 1959, 0, "Lynch syndrome - benign, reviewed by expert panel"),
    ("MSH2_ben", "NM_000251.3", 1666, 0, "Lynch syndrome - benign, reviewed by expert panel"),
    ("LDLR_ben", "NM_000527.5", 507, 0, "Familial hypercholesterolemia - benign, reviewed by expert panel"),
]

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

all_results = []

for mname, use_rna in [("codonbert",False),("codonbert_hf",True),("encodon-80m",False)]:
    print(f"\n{'='*60}\n{mname}\n{'='*60}")
    model, tokenizer, meta = CodonModelLoader.load(mname, device=DEVICE)
    if model is None: continue

    train_syn = syn.sample(min(5000,len(syn)), random_state=42)
    train_seqs, train_labs = [], []
    for _, row in train_syn.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
        if s is None: continue
        if use_rna: s = dna_to_rna(s)
        train_seqs.append(s); train_labs.append(row["label"])
    train_labs = np.array(train_labs, dtype=int)
    train_emb = extract_embeddings(model, tokenizer, train_seqs, device=DEVICE, batch_size=16, show_progress=True)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(train_emb, train_labs)

    for case_name, tx_id, cpos, true_label, description in cases:
        cds = cds_cache.get(tx_id, "")
        if not cds: continue
        codon_str = build_codon_str(cds, cpos)
        if codon_str is None: continue
        if use_rna: codon_str = dna_to_rna(codon_str)
        emb = extract_embeddings(model, tokenizer, [codon_str], device=DEVICE, batch_size=1, show_progress=False)
        prob = lr.predict_proba(emb)[0][1]
        pred = 1 if prob > 0.5 else 0
        correct = pred == true_label
        r = {"model": mname, "case": case_name, "tx_id": tx_id, "cpos": cpos, "true_label": true_label, "prob_pathogenic": round(float(prob),4), "predicted": pred, "correct": correct, "description": description}
        all_results.append(r)
        status = "✓" if correct else "✗"
        print(f"  {status} {case_name:15s} | true={true_label} | P(path)={prob:.4f} | {description[:40]}")

    CodonModelLoader.release(model, DEVICE)

with open(OUT_DIR/"case_study_results.json","w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for mname in ["codonbert","codonbert_hf","encodon-80m"]:
    mr = [r for r in all_results if r["model"]==mname]
    if not mr: continue
    n_correct = sum(r["correct"] for r in mr)
    path_cases = [r for r in mr if r["true_label"]==1]
    ben_cases = [r for r in mr if r["true_label"]==0]
    print(f"  {mname}: {n_correct}/{len(mr)} correct ({n_correct/len(mr)*100:.0f}%)")
    print(f"    Pathogenic: mean P={np.mean([r['prob_pathogenic'] for r in path_cases]):.3f}")
    print(f"    Benign:     mean P={np.mean([r['prob_pathogenic'] for r in ben_cases]):.3f}")