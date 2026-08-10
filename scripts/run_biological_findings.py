import sys; sys.path.insert(0, ".")
import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats
from collections import Counter

DATA_DIR = Path("./data")
OUT_DIR = Path("./results/biological_findings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CODON_CONTEXT = 16

HUMAN_CODON_USAGE = {
    'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.13, 'TTG': 0.13,
    'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.40,
    'ATT': 0.36, 'ATC': 0.47, 'ATA': 0.17, 'ATG': 1.00,
    'GTT': 0.18, 'GTC': 0.22, 'GTA': 0.12, 'GTG': 0.48,
    'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.05,
    'CCT': 0.28, 'CCC': 0.32, 'CCA': 0.27, 'CCG': 0.13,
    'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
    'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.27, 'GCG': 0.07,
    'TAT': 0.43, 'TAC': 0.57, 'TAA': 0.61, 'TAG': 0.39,
    'CAT': 0.42, 'CAC': 0.58, 'CAA': 0.73, 'CAG': 0.27,
    'AAT': 0.47, 'AAC': 0.53, 'AAA': 0.73, 'AAG': 0.27,
    'GAT': 0.46, 'GAC': 0.54, 'GAA': 0.58, 'GAG': 0.42,
    'TGT': 0.45, 'TGC': 0.55, 'TGA': 1.00, 'TGG': 1.00,
    'CGT': 0.08, 'CGC': 0.18, 'CGA': 0.07, 'CGG': 0.12,
    'AGT': 0.27, 'AGC': 0.24, 'AGA': 0.21, 'AGG': 0.21,
    'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.27, 'GGG': 0.23,
}

SYNONYMOUS_FAMILIES = {
    'F': ['TTT', 'TTC'], 'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'I': ['ATT', 'ATC', 'ATA'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], 'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'Y': ['TAT', 'TAC'],
    'H': ['CAT', 'CAC'], 'Q': ['CAA', 'CAG'], 'N': ['AAT', 'AAC'],
    'K': ['AAA', 'AAG'], 'D': ['GAT', 'GAC'], 'E': ['GAA', 'GAG'],
    'C': ['TGT', 'TGC'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
}

CODON_TO_AA = {}
for aa, codons in SYNONYMOUS_FAMILIES.items():
    for c in codons:
        CODON_TO_AA[c] = aa


def compute_cai(codons):
    if not codons:
        return 0.0
    from math import log, exp
    cai_sum = 0.0
    count = 0
    for c in codons:
        if c in CODON_TO_AA:
            aa = CODON_TO_AA[c]
            family = SYNONYMOUS_FAMILIES[aa]
            max_freq = max(HUMAN_CODON_USAGE.get(x, 0.001) for x in family)
            freq = HUMAN_CODON_USAGE.get(c, 0.001)
            if max_freq > 0:
                cai_sum += log(freq / max_freq)
                count += 1
    if count == 0:
        return 0.0
    return exp(cai_sum / count)


def compute_gc3(codons):
    if not codons:
        return 0.0
    return sum(1 for c in codons if len(c) == 3 and c[2] in "GC") / len(codons)


def dna_to_codons(seq):
    return [seq[i:i+3] for i in range(0, len(seq) - len(seq) % 3, 3)]


def parse_hgvs(name):
    name = str(name)
    m = re.match(r"(NM_\d+\.\d+)(?:\(\w+\))?:(c\.(\d+)([ACGT])>([ACGT]))", name)
    if m:
        return m.group(1), int(m.group(3)), m.group(4), m.group(5)
    return None, None, None, None


def build_cds_sequence(cds_seq, cpos, context_codons=CODON_CONTEXT):
    if cpos < 1 or cpos > len(cds_seq):
        return None
    codon_idx = (cpos - 1) // 3
    start_codon = max(0, codon_idx - context_codons)
    end_codon = min(len(cds_seq) // 3, codon_idx + context_codons + 1)
    codons = dna_to_codons(cds_seq[start_codon*3:end_codon*3])
    if not codons:
        return None
    return codons


def extract_attention_weights(model, tokenizer, sequences, device="cuda:2",
                               batch_size=4, max_length=2048, layer=-1):
    from src.eval.evaluation_utils import _fix_token_type_ids
    all_attentions = []
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = _fix_token_type_ids(tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True
        ).to(device))

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            except TypeError:
                try:
                    outputs = model(**inputs, output_attentions=True)
                except TypeError:
                    outputs = model(**inputs, output_hidden_states=True)

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden = outputs.hidden_states[layer]
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                all_embeddings.append(pooled.cpu().numpy())
            elif hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                all_embeddings.append(pooled.cpu().numpy())

            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                last_attn = outputs.attentions[-1]
                all_attentions.append(last_attn.cpu().numpy())

    embeddings = np.vstack(all_embeddings) if all_embeddings else None
    return embeddings, all_attentions


def analyze_attention_cai_correlation(attention_weights_list, codon_sequences_list,
                                       cai_values_list):
    results = []
    for batch_idx, attn_batch in enumerate(attention_weights_list):
        for seq_idx in range(attn_batch.shape[0]):
            n_heads = attn_batch.shape[1]
            seq_len = attn_batch.shape[2]

            for head_idx in range(n_heads):
                attn_map = attn_batch[seq_idx, head_idx, :, :]
                row_sums = attn_map.sum(axis=1)
                if seq_idx < len(cai_values_list):
                    cai = cai_values_list[seq_idx]
                    if len(row_sums) > 1 and not np.isnan(cai):
                        results.append({
                            "head": head_idx,
                            "attn_entropy": -np.sum(row_sums * np.log(row_sums + 1e-10)),
                            "cai": cai,
                        })
    return results


def embedding_attribute_regression(embeddings, attributes_dict):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings('ignore')

    results = {}
    for attr_name, attr_values in attributes_dict.items():
        attr_arr = np.array(attr_values)
        if len(attr_arr) != len(embeddings):
            continue
        valid = ~np.isnan(attr_arr)
        if valid.sum() < 50:
            continue
        X = embeddings[valid]
        y = attr_arr[valid]

        reg = LinearRegression()
        try:
            r2_scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
            results[attr_name] = {
                "r2_mean": round(float(r2_scores.mean()), 4),
                "r2_std": round(float(r2_scores.std()), 4),
                "n": int(valid.sum()),
            }
        except Exception:
            pass
    return results


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device = "cuda:2"

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
    task3_variants = variants[variants["is_synonymous"]].copy()

    n_patho = int((task3_variants["label"]==1).sum())
    n_benign = int((task3_variants["label"]==0).sum())
    max_samples = min(n_patho * 2, n_patho + n_benign, 2000)

    df = task3_variants.copy()
    if len(df) > max_samples:
        n_per = min(max_samples // 2, n_patho, n_benign)
        patho = df[df["label"]==1].sample(n=n_per, random_state=42)
        benign = df[df["label"]==0].sample(n=n_per, random_state=42)
        df = pd.concat([patho, benign]).sample(frac=1, random_state=42)

    records = []
    for _, row in df.iterrows():
        tx = row["tx_id"]
        cpos = int(row["cpos"])
        cds_seq = cds_cache.get(tx, "")
        if not cds_seq:
            continue
        codons = build_cds_sequence(cds_seq, cpos)
        if codons is None:
            continue
        records.append({
            "codons": codons,
            "codon_seq_str": " ".join(codons),
            "cpos": cpos,
            "cds_len": len(cds_seq),
            "label": row["label"],
            "cai": compute_cai(codons),
            "gc3": compute_gc3(codons),
        })

    print(f"  {len(records)} synonymous variant samples")

    sequences = [r["codon_seq_str"] for r in records]
    cai_values = [r["cai"] for r in records]
    gc3_values = [r["gc3"] for r in records]
    labels = [r["label"] for r in records]
    positions = [r["cpos"] / max(r["cds_len"], 1) for r in records]

    all_results = {}

    for clm_name in ["codonbert", "codonbert_hf", "encodon-80m"]:
        print(f"\n{'='*60}")
        print(f"Analyzing {clm_name}")
        print(f"{'='*60}")

        from src.models.loader import CodonModelLoader
        import src.models.xformers_compat

        model, tokenizer, meta = CodonModelLoader.load(clm_name, device=device)
        if model is None:
            print(f"  FAILED to load {clm_name}")
            continue

        print(f"  Extracting embeddings + attention...")
        embeddings, attn_weights = extract_attention_weights(
            model, tokenizer, sequences, device=device, batch_size=4
        )

        CodonModelLoader.release(model, device)

        if embeddings is not None:
            print(f"  Embeddings: {embeddings.shape}")

            # === Analysis 1: Embedding → Codon Attribute Regression ===
            print(f"\n  [Analysis 1] Embedding → Codon Attribute Regression")
            attributes = {
                "CAI": cai_values,
                "GC3": gc3_values,
                "position_norm": positions,
                "label_pathogenic": labels,
            }
            attr_results = embedding_attribute_regression(embeddings, attributes)
            for attr_name, res in attr_results.items():
                print(f"    {attr_name}: R²={res['r2_mean']:.4f}±{res['r2_std']:.4f} (n={res['n']})")

            # === Analysis 2: Pathogenic vs Benign embedding separation ===
            print(f"\n  [Analysis 2] Pathogenic vs Benign embedding analysis")
            labels_arr = np.array(labels)
            patho_emb = embeddings[labels_arr == 1]
            benign_emb = embeddings[labels_arr == 0]

            centroid_p = patho_emb.mean(axis=0)
            centroid_b = benign_emb.mean(axis=0)
            sep = np.linalg.norm(centroid_p - centroid_b)
            print(f"    Centroid separation: {sep:.4f}")

            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import StratifiedKFold
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = cross_val_score(clf, embeddings, labels_arr, cv=cv, scoring="roc_auc")
            print(f"    LR AUC: {aucs.mean():.4f}±{aucs.std():.4f}")

            # === Analysis 3: CAI correlation with pathogenicity ===
            print(f"\n  [Analysis 3] CAI vs pathogenicity")
            cai_arr = np.array(cai_values)
            patho_cai = cai_arr[labels_arr == 1]
            benign_cai = cai_arr[labels_arr == 0]
            t_stat, p_val = stats.ttest_ind(patho_cai, benign_cai)
            spearman_r, spearman_p = stats.spearmanr(cai_arr, labels_arr)
            print(f"    Pathogenic CAI: {patho_cai.mean():.4f}±{patho_cai.std():.4f}")
            print(f"    Benign CAI: {benign_cai.mean():.4f}±{benign_cai.std():.4f}")
            print(f"    t-test: t={t_stat:.3f}, p={p_val:.4f}")
            print(f"    Spearman(CAI, label): ρ={spearman_r:.4f}, p={spearman_p:.4f}")

            # === Analysis 4: Attention → CAI correlation ===
            print(f"\n  [Analysis 4] Attention → CAI correlation")
            attn_cai_results = []
            if attn_weights:
                for batch_attn in attn_weights:
                    for seq_idx in range(batch_attn.shape[0]):
                        global_idx = len(attn_cai_results)
                        if global_idx >= len(records):
                            break
                        n_heads = batch_attn.shape[1]
                        for h in range(n_heads):
                            attn_map = batch_attn[seq_idx, h, :, :]
                            diag_attn = np.diag(attn_map)
                            attn_entropy = -np.sum(
                                attn_map.mean(axis=0) * np.log(attn_map.mean(axis=0) + 1e-10)
                            )
                            attn_cai_results.append({
                                "head": h,
                                "attn_entropy": float(attn_entropy),
                                "cai": records[global_idx]["cai"],
                                "gc3": records[global_idx]["gc3"],
                                "label": records[global_idx]["label"],
                            })

            if attn_cai_results:
                attn_df = pd.DataFrame(attn_cai_results)
                for attr in ["cai", "gc3"]:
                    rho, p = stats.spearmanr(attn_df["attn_entropy"], attn_df[attr])
                    print(f"    Spearman(attn_entropy, {attr}): ρ={rho:.4f}, p={p:.4f}")

                for h in range(min(12, attn_df["head"].max() + 1)):
                    head_data = attn_df[attn_df["head"] == h]
                    if len(head_data) > 10:
                        rho, p = stats.spearmanr(head_data["attn_entropy"], head_data["cai"])
                        if abs(rho) > 0.05:
                            print(f"    Head {h}: Spearman(attn, CAI) ρ={rho:.4f}, p={p:.4f}")

            all_results[clm_name] = {
                "embedding_shape": list(embeddings.shape),
                "attribute_regression": attr_results,
                "centroid_separation": float(sep),
                "lr_auc": round(float(aucs.mean()), 4),
                "lr_auc_std": round(float(aucs.std()), 4),
                "cai_pathogenicity": {
                    "patho_cai_mean": round(float(patho_cai.mean()), 4),
                    "benign_cai_mean": round(float(benign_cai.mean()), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 6),
                },
            }

    with open(OUT_DIR / "biological_findings_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Biological findings analysis complete!")
    print(f"{'='*60}")
    for clm_name, res in all_results.items():
        print(f"\n{clm_name}:")
        print(f"  Centroid separation: {res['centroid_separation']:.4f}")
        print(f"  LR AUC: {res['lr_auc']:.4f}±{res['lr_auc_std']:.4f}")
        print(f"  CAI-pathogenicity: ρ={res['cai_pathogenicity']['spearman_r']:.4f}")
        for attr, ar in res.get("attribute_regression", {}).items():
            print(f"  Embedding→{attr}: R²={ar['r2_mean']:.4f}")


if __name__ == "__main__":
    main()