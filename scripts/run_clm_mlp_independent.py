import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, time, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

DEVICE = "cuda:0"; DATA_DIR = Path("./data"); OUT_DIR = Path("./results/supplementary"); OUT_DIR.mkdir(parents=True, exist_ok=True); CODON_CONTEXT = 16
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

def load_data():
    cds=json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
    raw=pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",low_memory=False)
    snv=raw[raw["Type"]=="single nucleotide variant"].copy()
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
    h=v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds.keys()))
    var=v[h].copy()
    var["is_syn"]=var["Name"].apply(lambda n:bool(re.search(r"p\.\w+\d+=",str(n))))
    return cds, var

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df["label"].sum()),int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n,random_state=42),df[df["label"]==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

def main():
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import extract_embeddings
    
    cds_cache, variants = load_data()
    t2=variants[~variants["is_syn"]].copy()
    t3=variants[variants["is_syn"]].copy()
    np3=int((t3["label"]==1).sum()); nb3=int((t3["label"]==0).sum())
    max3=min(np3*2,np3+nb3)
    
    all_results = []
    
    for mname, use_rna in [("codonbert",False),("codonbert_hf",True),("encodon-80m",False)]:
        print(f"\n{'='*60}\n{mname}\n{'='*60}")
        config = CodonModelLoader.get_config(mname)
        model, tokenizer, meta = CodonModelLoader.load(mname, device=DEVICE)
        if model is None: continue
        
        for task_name, task_df, max_n in [("task2_missense",t2,5000),("task3_synonymous",t3,max3)]:
            df = balance(task_df.copy(), max_n)
            seqs, labs = [], []
            for _, row in df.iterrows():
                s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
                if s is None: continue
                if use_rna: s = dna_to_rna(s)
                seqs.append(s); labs.append(row["label"])
            labs = np.array(labs, dtype=int)
            print(f"  {task_name}: {len(seqs)} seqs")
            
            emb = extract_embeddings(model, tokenizer, seqs, device=DEVICE, batch_size=16, show_progress=True)
            
            # Save embeddings for reuse
            np.save(OUT_DIR/f"{mname}_{task_name}_emb.npy", emb)
            np.save(OUT_DIR/f"{mname}_{task_name}_labels.npy", labs)
            
            # 80/20 split
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(emb, labs, test_size=0.2, random_state=42, stratify=labs)
            
            # LR
            lr = LogisticRegression(max_iter=2000,C=1.0,solver="lbfgs").fit(X_tr, y_tr)
            lr_test = roc_auc_score(y_te, lr.predict_proba(X_te)[:,1])
            lr_cv = cross_val_score(LogisticRegression(max_iter=2000,C=1.0,solver="lbfgs"), emb, labs, cv=StratifiedKFold(5,shuffle=True,random_state=42), scoring="roc_auc")
            
            # MLP (train once on 80%, test on 20%)
            mlp = MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=42,early_stopping=True,validation_fraction=0.1).fit(X_tr, y_tr)
            mlp_test = roc_auc_score(y_te, mlp.predict_proba(X_te)[:,1])
            
            r = {
                "model": mname, "task": task_name, "n": len(seqs),
                "cv_lr_mean": round(float(lr_cv.mean()),4), "cv_lr_std": round(float(lr_cv.std()),4),
                "test_lr": round(float(lr_test),4),
                "test_mlp": round(float(mlp_test),4),
                "n_train": len(X_tr), "n_test": len(X_te),
            }
            all_results.append(r)
            print(f"  CV-LR={r['cv_lr_mean']:.4f} | Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")
            
            with open(OUT_DIR/"cLM_mlp_independent_results.json","w") as f:
                json.dump(all_results, f, indent=2, default=str)
        
        CodonModelLoader.release(model, DEVICE)
    
    print(f"\n{'='*60}\nFINAL RESULTS\n{'='*60}")
    for r in all_results:
        print(f"  {r['model']:20s} | {r['task']:25s} | Test-LR={r['test_lr']:.4f} | Test-MLP={r['test_mlp']:.4f}")

if __name__=="__main__":
    main()
