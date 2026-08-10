"""Oracle probe: upper bound on extractable synonymous signal using all non-neural codon features + GBT."""
import sys; sys.path.insert(0, ".")
import json, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/oracle_probe")
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

def parse_hgvs(name):
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", str(name))
    return (m.group(1), int(m.group(3))) if m else (None, None)

def codon_frequency_features(cds_seq):
    codons = dna_to_codons(cds_seq)
    freq = np.zeros(64)
    codon_list = sorted(CODON_TABLE.keys())
    codon_idx = {c: i for i, c in enumerate(codon_list)}
    for c in codons:
        if c in codon_idx:
            freq[codon_idx[c]] += 1
    if freq.sum() > 0:
        freq /= freq.sum()
    return freq

def gc3_feature(cds_seq):
    codons = dna_to_codons(cds_seq)
    if not codons: return 0.0
    gc3 = sum(1 for c in codons if c[-1] in 'GC') / len(codons)
    return gc3

def position_codon_features(cds_seq, cpos, ctx=CODON_CONTEXT):
    cpos = int(cpos)
    if cpos < 1 or cpos > len(cds_seq): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds_seq) // 3, ci + ctx + 1)
    codons = dna_to_codons(cds_seq[s*3:e*3])
    if not codons: return None
    codon_list = sorted(CODON_TABLE.keys())
    codon_idx = {c: i for i, c in enumerate(codon_list)}
    n_pos = 2 * ctx + 1
    feat = np.zeros(n_pos * 64)
    for i, c in enumerate(codons):
        if c in codon_idx:
            feat[i * 64 + codon_idx[c]] = 1.0
    return feat

def kmer_features(cds_seq, cpos, k=6, ctx=CODON_CONTEXT):
    cpos = int(cpos)
    if cpos < 1 or cpos > len(cds_seq): return None
    ci = (cpos - 1) // 3
    s, e = max(0, ci - ctx), min(len(cds_seq) // 3, ci + ctx + 1)
    codons = dna_to_codons(cds_seq[s*3:e*3])
    if len(codons) < k: return None
    from collections import Counter
    kmers = []
    for i in range(len(codons) - k + 1):
        kmers.append("_".join(codons[i:i+k]))
    kmer_counts = Counter(kmers)
    all_kmers = sorted(set(kmers))
    kmer_idx = {km: i for i, km in enumerate(all_kmers)}
    feat = np.zeros(len(all_kmers))
    for km, cnt in kmer_counts.items():
        feat[kmer_idx[km]] = cnt
    if feat.sum() > 0:
        feat /= feat.sum()
    return feat

def load_task3(cds_cache, raw_data, max_n=2840):
    snv = raw_data[raw_data["Type"] == "single nucleotide variant"].copy()
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
    has = valid["tx_id"].notna() & valid["cpos"].notna() & valid["tx_id"].isin(set(cds_cache.keys()))
    variants = valid[has].copy()
    variants["is_syn"] = variants["Name"].apply(lambda n: bool(re.search(r"p\.\w+\d+=", str(n))))
    df = variants[variants["is_syn"]].copy()
    n_pc = min(max_n//2, int(df["label"].sum()), int(len(df)-df["label"].sum()))
    df = pd.concat([df[df["label"]==1].sample(n_pc, random_state=42),
                    df[df["label"]==0].sample(n_pc, random_state=42)])
    return df

if __name__ == "__main__":
    print("Loading data...")
    cds_cache = json.load(open(DATA_DIR / "task2_clinvar" / "cds_sequences.json"))
    raw = pd.read_csv(DATA_DIR / "task2_clinvar" / "clinvar_raw.txt.gz", sep="\t", low_memory=False)
    df = load_task3(cds_cache, raw)
    print(f"Loaded {len(df)} synonymous variants")

    print("Extracting oracle features...")
    all_features = []
    all_labels = []
    for _, row in df.iterrows():
        cds = cds_cache[row["tx_id"]]
        cpos = row["cpos"]
        freq_feat = codon_frequency_features(cds)
        gc3 = gc3_feature(cds)
        pos_feat = position_codon_features(cds, cpos)
        if pos_feat is None: continue
        combined = np.concatenate([freq_feat, [gc3], pos_feat])
        all_features.append(combined)
        all_labels.append(row["label"])

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"Feature matrix: {X.shape}, labels: {y.shape}, pos: {y.sum()}, neg: {len(y)-y.sum()}")

    print("\n=== Oracle probe: GradientBoosting (5-fold CV) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                          random_state=42, subsample=0.8)
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
    print(f"  Mean AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    print("\n=== Oracle probe: Independent test set (80/20) ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                      random_state=42, subsample=0.8)
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, pred)
    print(f"  Test AUC = {auc_test:.4f}")

    results = {
        "task": "task3_synonymous",
        "feature_description": "codon_freq(64) + gc3(1) + position_codon_onehot(33*64=2112) = 2177 features",
        "n_features": X.shape[1],
        "n_samples": len(y),
        "gbt_5fold_auc_mean": round(np.mean(fold_aucs), 4),
        "gbt_5fold_auc_std": round(np.std(fold_aucs), 4),
        "gbt_5fold_auc_folds": [round(a, 4) for a in fold_aucs],
        "gbt_independent_test_auc": round(auc_test, 4),
    }
    with open(OUT_DIR / "oracle_probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR / 'oracle_probe_results.json'}")