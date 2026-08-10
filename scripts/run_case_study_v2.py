import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

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

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df["label"].sum()),int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n,random_state=42),df[df["label"]==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

cds_cache = json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
cols = ["Name","Type","ClinicalSignificance","ReviewStatus","GeneSymbol","ReferenceAlleleVCF","AlternateAlleleVCF"]
raw = pd.read_csv(DATA_DIR/"task2_clinvar"/"cds_sequences.json".replace("cds_sequences.json","../clinvar_raw.txt.gz"),sep="\t",usecols=cols,low_memory=False) if False else pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",usecols=cols,low_memory=False)
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

np3=int((syn["label"]==1).sum()); nb3=int((syn["label"]==0).sum())
max3=min(np3*2,np3+nb3)
train_df = balance(syn.copy(), max3)

cases = [
    ("PAH:c.1197A>T", "NM_000277.3", 1197, 1, "Phenylketonuria (PKU)"),
    ("ATM:c.3576G>A", "NM_000051.4", 3576, 1, "Ataxia-telangiectasia"),
    ("KCNQ1:c.1032G>A", "NM_000218.3", 1032, 1, "Long QT syndrome"),
    ("LDLR:c.1216C>A", "NM_000527.5", 1216, 1, "Familial hypercholesterolemia"),
    ("CFTR:c.2988G>A", "NM_000492.4", 2988, 1, "Cystic fibrosis"),
    ("BRCA2:c.9117G>A", "NM_000059.4", 9117, 1, "Hereditary breast/ovarian cancer"),
    ("MLH1:c.1038G>A", "NM_000249.4", 1038, 1, "Lynch syndrome"),
    ("MSH2:c.2634G>A", "NM_000251.3", 2634, 1, "Lynch syndrome"),
    ("APC:c.1956C>T", "NM_000038.6", 1956, 1, "Familial adenomatous polyposis"),
    ("F8:c.5217C>T", "NM_000132.4", 5217, 1, "Hemophilia A"),
    ("ACADVL:c.1038G>A", "NM_000018.4", 1038, 0, "VLCAD deficiency (benign)"),
    ("GCK:c.1285A>C", "NM_000162.5", 1285, 0, "MODY (benign)"),
    ("MLH1:c.1959G>T", "NM_000249.4", 1959, 0, "Lynch syndrome (benign)"),
    ("MSH2:c.1666T>C", "NM_000251.3", 1666, 0, "Lynch syndrome (benign)"),
    ("LDLR:c.507C>T", "NM_000527.5", 507, 0, "FH (benign)"),
]

import src.models.xformers_compat
from src.models.loader import CodonModelLoader
from src.eval.evaluation_utils import extract_embeddings

all_results = []

for mname, use_rna in [("codonbert",False),("codonbert_hf",True),("encodon-80m",False)]:
    print(f"\n{'='*60}\n{mname}\n{'='*60}")
    model, tokenizer, meta = CodonModelLoader.load(mname, device=DEVICE)
    if model is None: continue

    train_seqs, train_labs = [], []
    for _, row in train_df.iterrows():
        s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
        if s is None: continue
        if use_rna: s = dna_to_rna(s)
        train_seqs.append(s); train_labs.append(row["label"])
    train_labs = np.array(train_labs, dtype=int)
    print(f"  Train: {len(train_seqs)} seqs ({int(train_labs.sum())} path, {int(len(train_labs)-train_labs.sum())} ben)")
    train_emb = extract_embeddings(model, tokenizer, train_seqs, device=DEVICE, batch_size=16, show_progress=True)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs").fit(train_emb, train_labs)
    mlp = MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=42,early_stopping=True,validation_fraction=0.1).fit(train_emb, train_labs)

    for case_name, tx_id, cpos, true_label, description in cases:
        cds = cds_cache.get(tx_id, "")
        if not cds: continue
        codon_str = build_codon_str(cds, cpos)
        if codon_str is None: continue
        if use_rna: codon_str = dna_to_rna(codon_str)
        emb = extract_embeddings(model, tokenizer, [codon_str], device=DEVICE, batch_size=1, show_progress=False)
        lr_prob = lr.predict_proba(emb)[0][1]
        mlp_prob = mlp.predict_proba(emb)[0][1]
        lr_pred = 1 if lr_prob > 0.5 else 0
        mlp_pred = 1 if mlp_prob > 0.5 else 0
        r = {"model": mname, "case": case_name, "tx_id": tx_id, "cpos": cpos, "true_label": true_label, "lr_prob": round(float(lr_prob),4), "mlp_prob": round(float(mlp_prob),4), "lr_pred": lr_pred, "mlp_pred": mlp_pred, "lr_correct": lr_pred==true_label, "mlp_correct": mlp_pred==true_label, "description": description}
        all_results.append(r)
        lr_s = "✓" if lr_pred==true_label else "✗"
        mlp_s = "✓" if mlp_pred==true_label else "✗"
        print(f"  {case_name:20s} | true={true_label} | LR:{lr_s} P={lr_prob:.3f} | MLP:{mlp_s} P={mlp_prob:.3f} | {description}")

    CodonModelLoader.release(model, DEVICE)

with open(OUT_DIR/"case_study_results.json","w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for mname in ["codonbert","codonbert_hf","encodon-80m"]:
    mr = [r for r in all_results if r["model"]==mname]
    if not mr: continue
    lr_acc = sum(r["lr_correct"] for r in mr)/len(mr)
    mlp_acc = sum(r["mlp_correct"] for r in mr)/len(mr)
    path_lr = [r["lr_prob"] for r in mr if r["true_label"]==1]
    ben_lr = [r["lr_prob"] for r in mr if r["true_label"]==0]
    path_mlp = [r["mlp_prob"] for r in mr if r["true_label"]==1]
    ben_mlp = [r["mlp_prob"] for r in mr if r["true_label"]==0]
    print(f"  {mname}: LR acc={lr_acc:.0%}, MLP acc={mlp_acc:.0%}")
    print(f"    LR:  Path P={np.mean(path_lr):.3f}, Ben P={np.mean(ben_lr):.3f}")
    print(f"    MLP: Path P={np.mean(path_mlp):.3f}, Ben P={np.mean(ben_mlp):.3f}")