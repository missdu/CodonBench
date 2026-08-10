import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch, numpy as np, json, re, time, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import stats

DEVICE = "cuda:0"
DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODON_CONTEXT = 16

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]
def dna_to_rna(seq): return seq.replace("T", "U")
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds): return None
    ci = (cpos-1)//3; s,e = max(0,ci-ctx), min(len(cds)//3,ci+ctx+1)
    c = dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
    cds = json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"]=="single nucleotide variant"].copy()
    pk, bk = ["Pathogenic","Likely pathogenic"], ["Benign","Likely benign"]
    def classify(cs):
        cs=str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    v = snv[snv["label"]>=0].copy()
    v = v[(v["ReferenceAlleleVCF"].str.len()==1)&(v["AlternateAlleleVCF"].str.len()==1)]
    p = v["Name"].apply(parse_hgvs)
    v["tx_id"]=p.apply(lambda x:x[0]); v["cpos"]=p.apply(lambda x:x[1])
    h = v["tx_id"].notna()&v["cpos"].notna()&v["tx_id"].isin(set(cds.keys()))
    var = v[h].copy()
    var["is_syn"]=var["Name"].apply(lambda n:bool(re.search(r"p\.\w+\d+=",str(n))))
    return cds, var

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df["label"].sum()),int(len(df)-df["label"].sum()))
        return pd.concat([df[df["label"]==1].sample(n,random_state=42),df[df["label"]==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

def run_eval(emb, labels, task_name, model_name, extra_info=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for pname, clf in [("LR", LogisticRegression(max_iter=2000,C=1.0,solver="lbfgs")),
                        ("MLP", MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42))]:
        aucs = cross_val_score(clf, emb, labels, cv=cv, scoring="roc_auc")
        results[pname] = {"mean":round(float(aucs.mean()),4), "std":round(float(aucs.std()),4),
                          "folds":[round(float(a),4) for a in aucs]}
    
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labels, test_size=0.2, random_state=42, stratify=labels)
    test_results = {}
    for pname, Cls in [("LR", LogisticRegression), ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42))]:
        clf = Cls() if callable(Cls) and not isinstance(Cls, type) else Cls(max_iter=2000,C=1.0,solver="lbfgs") if Cls==LogisticRegression else Cls()
        if pname=="LR": clf=LogisticRegression(max_iter=2000,C=1.0,solver="lbfgs")
        else: clf=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict_proba(X_te)[:,1]
        test_results[pname] = round(float(roc_auc_score(y_te, y_pred)),4)
    
    r = {"model":model_name, "task":task_name, "n":len(emb), "cv":results, "test":test_results}
    if extra_info: r.update(extra_info)
    print(f"  {model_name} | {task_name} | CV-LR={results['LR']['mean']:.4f} CV-MLP={results['MLP']['mean']:.4f} | Test-LR={test_results['LR']:.4f} Test-MLP={test_results['MLP']:.4f}")
    return r

def main():
    cds_cache, variants = load_data()
    t2 = variants[~variants["is_syn"]].copy()
    t3 = variants[variants["is_syn"]].copy()
    np3 = int((t3["label"]==1).sum()); nb3 = int((t3["label"]==0).sum())
    max3 = min(np3*2, np3+nb3)
    
    all_results = []
    
    # ============ 1. Mistral-Codon-117M (SKIPPED - MoE index error) ============
    print("\nSkipping Mistral-Codon-117M (MoE embedding index error)")
    
    # ============ 2. ESM-2 with MLP probing ============
    print("\n" + "="*60 + "\nESM-2 with MLP probing\n" + "="*60)
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(DEVICE).eval()
    bc = alphabet.get_batch_converter()
    
    def build_protein(cds, cpos, ctx=CODON_CONTEXT):
        if cpos<1 or cpos>len(cds): return None
        ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
        codons=dna_to_codons(cds[s*3:e*3])
        if not codons: return None
        ct={'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
            'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
            'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
            'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
            'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
            'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
            'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
            'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
        p="".join(ct.get(c,'X') for c in codons).replace('*','')
        return p[:1022] if len(p)>=5 else None
    
    for task_name, task_df, max_n in [("task2_missense",t2,5000),("task3_synonymous",t3,max3)]:
        df = balance(task_df.copy(), max_n)
        seqs, labs = [], []
        for _, row in df.iterrows():
            p = build_protein(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
            if p: seqs.append(p); labs.append(row["label"])
        labs = np.array(labs, dtype=int)
        
        all_embs = []
        for i in range(0, len(seqs), 4):
            data = [(f"s{j}",s) for j,s in enumerate(seqs[i:i+4])]
            _,_,tokens = bc(data)
            with torch.no_grad():
                out = model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            all_embs.append(out["representations"][33][:,0,:].cpu().numpy())
        emb = np.vstack(all_embs)
        
        r = run_eval(emb, labs, task_name, "ESM-2-650M",
                     {"model_type":"pLM","params_M":650,"architecture":"ESM-2"})
        all_results.append(r)
    
    del model; torch.cuda.empty_cache()
    
    # ============ 3. Existing cLMs with MLP probing + independent test ============
    print("\n" + "="*60 + "\nExisting cLMs with MLP probing\n" + "="*60)
    import src.models.xformers_compat
    from src.models.loader import CodonModelLoader
    from src.eval.evaluation_utils import extract_embeddings
    
    for mname, use_rna in [("codonbert",False),("codonbert_hf",True),("encodon-80m",False)]:
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
            
            emb = extract_embeddings(model, tokenizer, seqs, device=DEVICE, batch_size=16, show_progress=True)
            r = run_eval(emb, labs, task_name, mname,
                         {"model_type":"cLM","params_M":config.params_M,"architecture":config.architecture})
            all_results.append(r)
        
        CodonModelLoader.release(model, DEVICE)
    
    # Save
    with open(OUT_DIR/"supplementary_results.json","w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}\nSUPPLEMENTARY RESULTS COMPLETE\n{'='*60}")
    for r in all_results:
        cv_mlp = r.get("cv",{}).get("MLP",{}).get("mean","N/A")
        test_mlp = r.get("test",{}).get("MLP","N/A")
        print(f"  {r['model']:25s} | {r['task']:25s} | CV-MLP={cv_mlp} | Test-MLP={test_mlp}")

if __name__=="__main__":
    main()