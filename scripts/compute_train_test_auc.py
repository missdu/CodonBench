import os; os.environ.pop("http_proxy", None); os.environ.pop("https_proxy", None)
import sys; sys.path.insert(0, ".")
import json, re, time
import numpy as np, pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import BertForMaskedLM, PreTrainedTokenizer

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "train_test_auc.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

CODON_VOCAB = [
    "TTT","TTC","TTA","TTG","TCT","TCC","TCA","TCG",
    "TAT","TAC","TAA","TAG","TGT","TGC","TGA","TGG",
    "CTT","CTC","CTA","CTG","CCT","CCC","CCA","CCG",
    "CAT","CAC","CAA","CAG","CGT","CGC","CGA","CGG",
    "ATT","ATC","ATA","ATG","ACT","ACC","ACA","ACG",
    "AAT","AAC","AAA","AAG","AGT","AGC","AGA","AGG",
    "GTT","GTC","GTA","GTG","GCT","GCC","GCA","GCG",
    "GAT","GAC","GAA","GAG","GGT","GGC","GGA","GGG",
]
SPECIAL_TOKENS = ["[PAD]","[UNK]","[CLS]","[SEP]","\n"]

class CodonTokenizer(PreTrainedTokenizer):
    vocab_size = len(SPECIAL_TOKENS) + len(CODON_VOCAB)
    def __init__(self, **kwargs):
        vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + CODON_VOCAB)}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 512)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", SPECIAL_TOKENS[4])
        super().__init__(**kwargs)
    def _tokenize(self, text):
        text = text.upper().strip()
        if " " in text: return text.split()
        return [text[i:i+3] for i in range(0, len(text)-2, 3)]
    def _convert_token_to_id(self, token): return self.vocab_dict.get(token, self.vocab_dict["[UNK]"])
    def _convert_id_to_token(self, index): return self.id_to_token.get(index, "[UNK]")
    def convert_tokens_to_string(self, tokens): return " ".join(tokens)
    def get_vocab(self): return self.vocab_dict
    @property
    def is_fast(self): return False
    def save_vocabulary(self, *a, **k): return []

class CharTokenizer(PreTrainedTokenizer):
    vocab_size = 9
    def __init__(self, **kwargs):
        vocab = {"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"\n":4,"A":5,"T":6,"G":7,"C":8}
        self.vocab_dict = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        kwargs["model_max_length"] = kwargs.get("model_max_length", 1538)
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "\n")
        super().__init__(**kwargs)
    def _tokenize(self, text):
        text = text.upper().strip().replace(" ", "")
        return list(text)
    def _convert_token_to_id(self, token): return self.vocab_dict.get(token, self.vocab_dict["[UNK]"])
    def _convert_id_to_token(self, index): return self.id_to_token.get(index, "[UNK]")
    def convert_tokens_to_string(self, tokens): return "".join(tokens)
    def get_vocab(self): return self.vocab_dict
    @property
    def is_fast(self): return False
    def save_vocabulary(self, *a, **k): return []

CODON_CONTEXT = 16

def dna_to_codons(seq): return [seq[i:i+3] for i in range(0, len(seq)-len(seq)%3, 3)]
def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:c\.(\d+)([ACGT])>([ACGT])", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)
def build_codon_str(cds, cpos, ctx=CODON_CONTEXT):
    if cpos<1 or cpos>len(cds): return None
    ci=(cpos-1)//3; s,e=max(0,ci-ctx),min(len(cds)//3,ci+ctx+1)
    c=dna_to_codons(cds[s*3:e*3])
    return " ".join(c) if c else None

def load_data():
    log("Loading data...")
    cds=json.load(open(DATA_DIR/"task2_clinvar"/"cds_sequences.json"))
    raw=pd.read_csv(DATA_DIR/"task2_clinvar"/"clinvar_raw.txt.gz",sep="\t",low_memory=False,
                     usecols=['Type','Name','ClinicalSignificance','ReferenceAlleleVCF','AlternateAlleleVCF'])
    snv=raw[raw['Type']=='single nucleotide variant'].copy()
    del raw
    pk,bk=['Pathogenic','Likely pathogenic'],['Benign','Likely benign']
    def classify(cs):
        cs=str(cs)
        if any(k in cs for k in pk): return 1
        if any(k in cs for k in bk): return 0
        return -1
    snv['label']=snv['ClinicalSignificance'].apply(classify)
    v=snv[snv['label']>=0].copy()
    del snv
    v=v[(v['ReferenceAlleleVCF'].str.len()==1)&(v['AlternateAlleleVCF'].str.len()==1)]
    pattern = re.compile(r"(NM_\d+\.\d+)(?:\(\w+\))?:c\.(\d+)([ACGT])>([ACGT])")
    matches = v['Name'].apply(lambda n: pattern.match(str(n)))
    v['tx_id'] = matches.apply(lambda m: m.group(1) if m else None)
    v['cpos'] = matches.apply(lambda m: int(m.group(2)) if m else None)
    h=v['tx_id'].notna()&v['cpos'].notna()&v['tx_id'].isin(set(cds.keys()))
    var=v[h].copy()
    del v
    var['is_syn']=var['Name'].apply(lambda n: bool(re.search(r'p\.\w+\d+=', str(n))))
    return cds, var

def balance(df, max_n):
    if max_n and len(df)>max_n:
        n=min(max_n//2,int(df['label'].sum()),int(len(df)-df['label'].sum()))
        return pd.concat([df[df['label']==1].sample(n,random_state=42),df[df['label']==0].sample(n,random_state=42)]).sample(frac=1,random_state=42)
    return df

def extract_embeddings(model, tokenizer, sequences, device='cpu', batch_size=32):
    model.eval()
    model.to(device)
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            mask = inputs['attention_mask'].unsqueeze(-1)
            emb = (last_hidden * mask).sum(1) / mask.sum(1)
            all_embeddings.append(emb.cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                log(f"    Extracted {i+len(batch)}/{len(sequences)} embeddings")
    return np.concatenate(all_embeddings, axis=0)

MODELS = {
    "char-v4": {
        "model_path": "./from_scratch_models/char-bert-ablation-v4-scale110m",
        "tokenizer_cls": CharTokenizer,
        "use_rna": False,
    },
    "codon-v4": {
        "model_path": "./from_scratch_models/codon-bert-ablation-v4-scale110m",
        "tokenizer_cls": CodonTokenizer,
        "use_rna": False,
    },
}

def main():
    log("=== Train vs Test AUC Analysis START ===")
    cds_cache, variants = load_data()
    
    for task_name, is_syn_task in [("task3_synonymous", True), ("task2_missense", False)]:
        log(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")
        
        if is_syn_task:
            subset = variants[variants["is_syn"]].copy()
        else:
            subset = variants.copy()
        
        np_ = int((subset['label']==1).sum())
        nb_ = int((subset['label']==0).sum())
        max_n = min(np_*2, np_+nb_)
        df = balance(subset.copy(), max_n)
        
        seqs, labs = [], []
        for _, row in df.iterrows():
            s = build_codon_str(cds_cache.get(row["tx_id"],""), int(row["cpos"]))
            if s is None: continue
            seqs.append(s)
            labs.append(row["label"])
        labels = np.array(labs, dtype=int)
        log(f"  {task_name}: {len(labels)} samples, {labels.sum()} pos, {len(labels)-labels.sum()} neg")
        
        for model_name, cfg in MODELS.items():
            log(f"\n  Model: {model_name}")
            log(f"    Loading model from {cfg['model_path']}...")
            tokenizer = cfg['tokenizer_cls']()
            model = BertForMaskedLM.from_pretrained(cfg['model_path'])
            log(f"    Model loaded, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
            
            log(f"    Extracting embeddings...")
            t0 = time.time()
            emb = extract_embeddings(model, tokenizer, seqs, device='cpu', batch_size=16)
            log(f"    Embeddings shape: {emb.shape}, took {time.time()-t0:.0f}s")
            del model
            
            # LR 5-fold CV (same as original)
            log(f"    Running LR 5-fold CV...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_fold_aucs = []
            lr_fold_train_aucs = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(emb, labels)):
                X_tr, X_te = emb[train_idx], emb[test_idx]
                y_tr, y_te = labels[train_idx], labels[test_idx]
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                lr = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
                lr.fit(X_tr_s, y_tr)
                train_auc = roc_auc_score(y_tr, lr.predict_proba(X_tr_s)[:, 1])
                test_auc = roc_auc_score(y_te, lr.predict_proba(X_te_s)[:, 1])
                lr_fold_aucs.append(test_auc)
                lr_fold_train_aucs.append(train_auc)
                log(f"      Fold {fold+1}: train_auc={train_auc:.4f}, test_auc={test_auc:.4f}")
            
            log(f"    LR test AUC: {np.mean(lr_fold_aucs):.4f} +/- {np.std(lr_fold_aucs):.4f}")
            log(f"    LR train AUC: {np.mean(lr_fold_train_aucs):.4f} +/- {np.std(lr_fold_train_aucs):.4f}")
            lr_overfit = np.mean(lr_fold_train_aucs) - np.mean(lr_fold_aucs)
            log(f"    LR overfit gap (train-test): {lr_overfit:.4f} ({lr_overfit*100:.1f} pp)")
            
            # MLP single split (same as original)
            log(f"    Running MLP train/test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                emb, labels, test_size=0.2, random_state=42, stratify=labels
            )
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42,
                               early_stopping=True, validation_fraction=0.1, solver='adam',
                               learning_rate_init=1e-3, alpha=1e-4)
            mlp.fit(X_train_s, y_train)
            mlp_train_auc = roc_auc_score(y_train, mlp.predict_proba(X_train_s)[:, 1])
            mlp_test_auc = roc_auc_score(y_test, mlp.predict_proba(X_test_s)[:, 1])
            mlp_overfit = mlp_train_auc - mlp_test_auc
            log(f"    MLP train AUC: {mlp_train_auc:.4f}")
            log(f"    MLP test AUC: {mlp_test_auc:.4f}")
            log(f"    MLP overfit gap: {mlp_overfit:.4f} ({mlp_overfit*100:.1f} pp)")
            
            # Save results
            result = {
                "model": model_name,
                "task": task_name,
                "n_samples": int(len(labels)),
                "lr_test_auc_mean": round(float(np.mean(lr_fold_aucs)), 4),
                "lr_test_auc_std": round(float(np.std(lr_fold_aucs)), 4),
                "lr_train_auc_mean": round(float(np.mean(lr_fold_train_aucs)), 4),
                "lr_train_auc_std": round(float(np.std(lr_fold_train_aucs)), 4),
                "lr_overfit_gap_pp": round(float(lr_overfit * 100), 1),
                "mlp_train_auc": round(float(mlp_train_auc), 4),
                "mlp_test_auc": round(float(mlp_test_auc), 4),
                "mlp_overfit_gap_pp": round(float(mlp_overfit * 100), 1),
            }
            
            res_file = OUT_DIR / "train_test_auc_results.json"
            existing = []
            if res_file.exists():
                existing = json.load(open(res_file))
            existing.append(result)
            with open(res_file, "w") as f:
                json.dump(existing, f, indent=2)
            log(f"    Saved result for {model_name} {task_name}")

    log("\n=== Train vs Test AUC Analysis DONE ===")

if __name__ == "__main__":
    main()