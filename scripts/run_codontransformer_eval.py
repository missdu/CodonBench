"""CodonTransformer评估 - 使用{aa}_{codon}格式输入"""
import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/codontransformer")
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


def codon_to_aa_codon_format(codon_seq):
    tokens = []
    for codon in codon_seq.split():
        aa = CODON_TABLE.get(codon.upper(), 'X')
        tokens.append(f"{aa.lower()}_{codon.lower()}")
    return " ".join(tokens)


def build_codon_context(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return " ".join(codons)


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = "cuda:2"
    HF_ID = "adibvafa/CodonTransformer"

    print("Loading CodonTransformer...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(HF_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_ID, trust_remote_code=True)
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify tokenization
    test_codon = "ATG GCT TTT GAA CCG"
    test_aa_codon = codon_to_aa_codon_format(test_codon)
    tokens = tokenizer(test_aa_codon, return_tensors="pt")
    decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    unk = sum(1 for t in decoded if t in ['[UNK]', '<unk>'])
    print(f"  Tokenization test: '{test_aa_codon}' -> UNK: {unk}/{len(decoded)} ({unk/max(len(decoded),1):.1%})")
    if unk / max(len(decoded), 1) > 0.1:
        print("  WARNING: High UNK ratio, CodonTransformer may not work properly")
        return

    print("Loading CDS cache...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))

    print("Loading ClinVar...")
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    snv = raw[raw["Type"] == "single nucleotide variant"].copy()

    patho_kw = ["Pathogenic", "Likely pathogenic"]
    benign_kw = ["Benign", "Likely benign"]
    def classify(cs):
        cs = str(cs)
        if any(k in cs for k in patho_kw): return 1
        if any(k in cs for k in benign_kw): return 0
        return -1

    snv["label"] = snv["ClinicalSignificance"].apply(classify)
    labeled = snv[snv["label"] >= 0].copy()
    valid = labeled[
        (labeled["ReferenceAlleleVCF"] != "na") &
        (labeled["AlternateAlleleVCF"] != "na") &
        (labeled["ReferenceAlleleVCF"].str.len() == 1) &
        (labeled["AlternateAlleleVCF"].str.len() == 1)
    ].copy()

    parsed = valid["Name"].apply(parse_hgvs)
    valid["tx_id"] = parsed.apply(lambda x: x[0])
    valid["cpos"] = parsed.apply(lambda x: x[1])

    has_all = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has_all].copy()

    def is_synonymous(name):
        return bool(re.search(r"p\.\w+\d+=", str(name)))

    variants["is_synonymous"] = variants["Name"].apply(is_synonymous)
    task2_variants = variants[~variants["is_synonymous"]].copy()
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_task3_patho = int((task3_variants["label"]==1).sum())
    n_task3_benign = int((task3_variants["label"]==0).sum())
    task3_max = min(n_task3_patho * 2, n_task3_patho + n_task3_benign)

    tasks = [
        ("task2_missense", task2_variants, 5000),
        ("task3_synonymous", task3_variants, task3_max),
    ]

    all_results = []

    for task_name, task_df, max_samples in tasks:
        print(f"\n{'='*60}")
        print(f"Evaluating CodonTransformer on {task_name}")
        print(f"{'='*60}")

        df = task_df.copy()
        if max_samples and len(df) > max_samples:
            n_per = min(max_samples // 2, int(df["label"].sum()), int(len(df) - df["label"].sum()))
            patho = df[df["label"]==1].sample(n=n_per, random_state=42)
            benign = df[df["label"]==0].sample(n=n_per, random_state=42)
            df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

        aa_codon_seqs = []
        labels = []
        skipped = 0

        for _, row in df.iterrows():
            tx = row["tx_id"]
            cpos = int(row["cpos"])
            cds_seq = cds_cache.get(tx, "")
            if not cds_seq:
                skipped += 1
                continue
            codon_seq = build_codon_context(cds_seq, cpos)
            if codon_seq is None:
                skipped += 1
                continue
            aa_codon_seqs.append(codon_to_aa_codon_format(codon_seq))
            labels.append(row["label"])

        labels = np.array(labels, dtype=int)
        print(f"  {len(aa_codon_seqs)} samples (skipped {skipped})")

        if len(aa_codon_seqs) < 100:
            all_results.append({"model": "CodonTransformer", "task": task_name, "success": False})
            continue

        # Extract embeddings
        print(f"  Extracting embeddings...")
        t0 = time.time()
        all_embs = []
        for i in range(0, len(aa_codon_seqs), 8):
            batch = aa_codon_seqs[i:i+8]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                              max_length=512, padding=True).to(device)
            with torch.no_grad():
                try:
                    out = model(**inputs, output_hidden_states=True)
                    hidden = out.hidden_states[-2]
                except TypeError:
                    out = model(**inputs)
                    hidden = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embs.append(pooled.cpu().numpy())
        embeddings = np.vstack(all_embs)
        emb_time = time.time() - t0
        print(f"  Embeddings: {embeddings.shape}, time={emb_time:.1f}s")

        # LR
        clf_lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs_lr = cross_val_score(clf_lr, embeddings, labels, cv=cv, scoring="roc_auc")
        print(f"  LR AUC = {aucs_lr.mean():.4f} ± {aucs_lr.std():.4f}")

        # MLP
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42, stratify=labels
        )
        clf_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                                early_stopping=True, random_state=42)
        clf_mlp.fit(X_train, y_train)
        mlp_auc = roc_auc_score(y_test, clf_mlp.predict_proba(X_test)[:, 1])
        print(f"  MLP AUC = {mlp_auc:.4f}")

        result = {
            "model": "CodonTransformer", "task": task_name, "type": "cLM (aa-codon)",
            "emb_dim": embeddings.shape[1], "n_samples": len(aa_codon_seqs),
            "success": True,
            "lr_auc_mean": round(float(aucs_lr.mean()), 4),
            "lr_auc_std": round(float(aucs_lr.std()), 4),
            "mlp_auc_test": round(float(mlp_auc), 4),
            "emb_time_s": round(emb_time, 1),
        }
        all_results.append(result)

    del model
    torch.cuda.empty_cache()

    with open(OUT_DIR / "codontransformer_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCodonTransformer evaluation complete!")
    for r in all_results:
        if r.get("success"):
            print(f"  {r['task']:25s} | LR={r['lr_auc_mean']:.4f}±{r['lr_auc_std']:.4f} | MLP={r.get('mlp_auc_test','N/A')}")


if __name__ == "__main__":
    main()