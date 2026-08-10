"""P0-2 extended: Compute channel-specificity evidence summary.

Key question: How strong is the evidence that probing-depth dependency
is channel-specific, given current data + pending ESM-2 MLP results?
"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("./results")

def main():
    # Load protocol sensitivity results
    with open(RESULTS_DIR / "protocol_sensitivity.json") as f:
        sens = json.load(f)

    print("=" * 70)
    print("CHANNEL-SPECIFICITY EVIDENCE SUMMARY")
    print("=" * 70)

    # 1. Protocol sensitivity (Spearman)
    print("\n1. PROTOCOL SENSITIVITY (Spearman ρ)")
    mis_lr_mlp = sens["task2_missense"]["lr_vs_mlp"]
    syn_lr_mlp = sens["task3_synonymous"]["lr_vs_mlp"]
    print(f"   MisPath LR↔MLP: ρ={mis_lr_mlp['spearman_rho']:.3f}, p={mis_lr_mlp['p_value']:.4f}")
    print(f"   SynPath LR↔MLP: ρ={syn_lr_mlp['spearman_rho']:.3f}, p={syn_lr_mlp['p_value']:.4f}")
    print(f"   → Δρ = {mis_lr_mlp['spearman_rho'] - syn_lr_mlp['spearman_rho']:.3f} (channel-specificity gap)")

    # 2. LR→MLP gains by model type
    print("\n2. LR→MLP GAINS BY MODEL TYPE")
    
    # cLMs with codon tokenization
    clm_codon = {
        "CodonBERT": {"mis_lr": 0.660, "mis_mlp": 0.677, "syn_lr": 0.716, "syn_mlp": 0.849},
        "CodonBERT-HF": {"mis_lr": 0.646, "mis_mlp": 0.650, "syn_lr": 0.668, "syn_mlp": 0.816},
        "EnCodon-80M": {"mis_lr": 0.636, "mis_mlp": 0.674, "syn_lr": 0.687, "syn_mlp": 0.812},
        "EnCodon-620M": {"mis_lr": 0.617, "mis_mlp": 0.667, "syn_lr": 0.785, "syn_mlp": 0.800},
        "CodonTransformer": {"mis_lr": 0.694, "mis_mlp": 0.680, "syn_lr": 0.713, "syn_mlp": 0.818},
    }
    
    # cLMs with non-codon tokenization
    clm_noncodon = {
        "CaLM": {"mis_lr": 0.689, "mis_mlp": 0.692, "syn_lr": 0.673, "syn_mlp": 0.783},
        "cdsBERT (char)": {"mis_lr": 0.629, "mis_mlp": 0.609, "syn_lr": 0.598, "syn_mlp": 0.582},
        "Mistral-16M": {"mis_lr": 0.671, "mis_mlp": 0.676, "syn_lr": 0.598, "syn_mlp": 0.841},
        "Mistral-1M": {"mis_lr": 0.666, "mis_mlp": 0.671, "syn_lr": 0.591, "syn_mlp": 0.734},
        "Mistral-117M": {"mis_lr": 0.611, "mis_mlp": 0.574, "syn_lr": 0.656, "syn_mlp": 0.800},
    }
    
    # pLMs (LR only for now)
    plms = {
        "ESM-2-650M": {"mis_lr": 0.719, "syn_lr": 0.680},
        "ESM-1b-650M": {"mis_lr": 0.711, "syn_lr": 0.600},
    }

    print("\n   Codon-tokenized cLMs:")
    for name, d in clm_codon.items():
        mis_gain = d["mis_mlp"] - d["mis_lr"]
        syn_gain = d["syn_mlp"] - d["syn_lr"]
        print(f"     {name:20s}: MisPath {d['mis_lr']:.3f}→{d['mis_mlp']:.3f} ({mis_gain:+.3f}) | SynPath {d['syn_lr']:.3f}→{d['syn_mlp']:.3f} ({syn_gain:+.3f})")
    
    avg_mis_codon = np.mean([d["mis_mlp"] - d["mis_lr"] for d in clm_codon.values()])
    avg_syn_codon = np.mean([d["syn_mlp"] - d["syn_lr"] for d in clm_codon.values()])
    print(f"     {'AVERAGE':20s}: MisPath gain={avg_mis_codon:+.3f} | SynPath gain={avg_syn_codon:+.3f}")

    print("\n   Non-codon cLMs:")
    for name, d in clm_noncodon.items():
        mis_gain = d["mis_mlp"] - d["mis_lr"]
        syn_gain = d["syn_mlp"] - d["syn_lr"]
        print(f"     {name:20s}: MisPath {d['mis_lr']:.3f}→{d['mis_mlp']:.3f} ({mis_gain:+.3f}) | SynPath {d['syn_lr']:.3f}→{d['syn_mlp']:.3f} ({syn_gain:+.3f})")

    print("\n   pLMs (LR only, MLP pending):")
    for name, d in plms.items():
        print(f"     {name:20s}: MisPath LR={d['mis_lr']:.3f} | SynPath LR={d['syn_lr']:.3f}")

    # 3. From-scratch ablation evidence
    print("\n3. FROM-SCRATCH ABLATION EVIDENCE")
    ablation = {
        "v1 (4.8K CDS, 10ep)": {"codon_mlp_syn": 0.719, "char_mlp_syn": 0.627, "delta": +0.092},
        "v3a (4.8K CDS, 100ep)": {"codon_mlp_syn": 0.720, "char_mlp_syn": 0.606, "delta": +0.114},
        "v3b (114K CDS, 30ep)": {"codon_mlp_syn": 0.792, "char_mlp_syn": 0.647, "delta": +0.145},
    }
    for name, d in ablation.items():
        print(f"   {name}: Codon MLP={d['codon_mlp_syn']:.3f}, Char MLP={d['char_mlp_syn']:.3f}, Δ={d['delta']:+.3f}")

    # 4. Scenario analysis for ESM-2 MLP
    print("\n4. SCENARIO ANALYSIS: ESM-2 MLP on SynPath")
    print("   Current ESM-2 LR: MisPath=0.719, SynPath=0.680")
    print()
    scenarios = [
        ("Best case", 0.680, "No gain → pLMs cannot access σ channel at any depth"),
        ("Moderate", 0.720, "Small gain (+4pp) → σ channel partially accessible to pLMs"),
        ("Worst case", 0.800, "Large gain (+12pp) → σ channel fully accessible → claim collapses"),
    ]
    for name, mlp_val, interpretation in scenarios:
        gain = mlp_val - 0.680
        codon_avg_gain = avg_syn_codon
        ratio = gain / codon_avg_gain if codon_avg_gain > 0 else float('inf')
        print(f"   {name:15s}: ESM-2 MLP={mlp_val:.3f} (gain={gain:+.3f})")
        print(f"                   cLM avg gain={codon_avg_gain:+.3f}, ratio={ratio:.2f}")
        print(f"                   → {interpretation}")
        print()

    # 5. Evidence chain strength
    print("5. EVIDENCE CHAIN STRENGTH (current)")
    print("   C1: Codon tokenization = correct inductive bias")
    print("     - From-scratch ablation: ✅✅✅ (3 versions, Δ=+9~14pp)")
    print("     - k-mer gradient: ❌ (not yet tested)")
    print("     - mRNABERT dual tokenizer: ❌ (not yet tested)")
    print()
    print("   C2: Inductive bias encodes σ at nonlinear depth")
    print("     - LR-blind/MLP-visible: ✅✅✅ (from-scratch + production models)")
    print("     - ESM-2 MLP: ❓ (PENDING - critical)")
    print()
    print("   C3: Depth-dependency is channel-specific")
    print("     - Protocol sensitivity: ✅✅✅ (ρ_Mis=0.855**, ρ_Syn=0.588 n.s.)")
    print("     - ESM-2 MLP: ❓ (PENDING - if ESM-2 MLP SynPath ≈ LR, strongest evidence)")
    print()
    print("   C4: Single-tier evaluation is a biased evaluator")
    print("     - Ranking instability: ✅✅ (SynPath ρ=0.588)")
    print("     - CaLM example: ✅ (0.673→0.783)")
    print()
    print("   C5: Codon > BPE > Char gradient")
    print("     - NT-v2-500M 0.699: ✅ (one data point)")
    print("     - k-mer ablation: ❌ (not yet tested)")

if __name__ == "__main__":
    main()