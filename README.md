# CodonBench: Leakage-Controlled Benchmarking of Codon Language Models

A benchmark and evaluation framework for codon language models (cLMs) with five leakage controls built in by default.

> **A thin synonymous channel: leakage and probe choices inflate reported gains of codon language models**

## Overview

CodonBench evaluates 21 models (11 cLMs, 2 protein LMs, 2 DNA LMs, 6 traditional baselines) on 6 tasks spanning the synonymous (SynPath) and missense (MisPath) channels, using a four-level probing hierarchy (zero-shot LLR → linear probe → nonlinear MLP → LoRA fine-tuning).

The framework implements five evaluation controls:

1. **Gene-held-out splitting** (LOGO-CV) — prevents gene-identity leakage
2. **Memorization-baseline canary** — detects whether a task is decidable without learned representations
3. **Probe-depth ladder** — separates tokenization effects from probe-capacity effects
4. **Single-variable probe ablation** (M0→M5) — decomposes apparent gains into defensible defaults
5. **Best-epoch selection audit** — prevents test-set selection bias

## Repository Structure

```
CodonBench/
├── src/                    # Core library
│   ├── data/               # Data loading and CDS retrieval
│   ├── models/             # Unified model loading + adapters
│   ├── eval/               # LLR computation, embedding extraction, probing
│   ├── agent/              # CodonBench-Agent orchestration layer
│   ├── stats/              # Statistical utilities
│   └── viz/                # Visualization helpers
├── scripts/                # Experiment scripts (run on server)
├── configs/                # Model and task configurations
│   ├── models.yaml         # 21 model definitions
│   └── tasks.yaml          # 6 task definitions
└── results/                # Benchmark results (JSON)
```

## Installation

```bash
git clone https://github.com/missdu/CodonBench.git
cd CodonBench
pip install -r requirements.txt
```

Requires Python 3.11+, PyTorch 2.0+, and a CUDA GPU.

## Quick Start

```python
from src.agent.codonbench_agent import CodonBenchAgent

agent = CodonBenchAgent()
results = agent.run(
    model="esm2_650m",
    task="synpath",
    split="logo_cv",       # gene-held-out
    probe="mlp",
    controls="all",         # enable all five controls
)
```

If a pre-flight check fails (e.g., gene overlap detected), the agent raises a diagnostic error rather than proceeding with a compromised evaluation.

## Models Evaluated

| Category | Models |
|----------|--------|
| cLMs (11) | EnCodon-620M/80M, CodonBERT, CodonBERT-HF, CodonTransformer, CaLM, Mistral-Codon (3 sizes), cdsBERT, cdsBERT-plus |
| Protein LMs (2) | ESM-2-650M, ESM-1b |
| DNA LMs (2) | Nucleotide Transformer v2-500M/50M |
| Baselines (6) | onehot-pos, onehot-freq, kmer3, kmer4, kmer6, combined |

Plus AlphaMissense as a specialized external reference.

## Tasks

| Task | Name | Type | n | Channel |
|------|------|------|---|---------|
| MisPath | Missense pathogenicity | Classification | 5,000 | I(A;Y) |
| SynPath | Synonymous pathogenicity | Classification | 2,840 | I(σ;Y\|A) |
| mRFPExpr | mRFP within-protein expression | Regression | 1,459 | I(σ;Y\|A) |
| EcoliExpr | E. coli cross-protein expression | Regression | 3,000 | I(A;Y) |
| mRNAStab | mRNA stability | Regression | 5,000 | Dual |
| FungalExpr | Fungal cross-protein expression | Regression | 7,089 | I(σ;Y\|A) |

## Data Sources

- **ClinVar** (MisPath, SynPath): Publicly available from NCBI
- **CodonBERT benchmark** (mRFPExpr, EcoliExpr, mRNAStab): [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT)
- **Fungal expression** (FungalExpr): From CodonBERT benchmark; Wint et al. 2022

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{liang2026codonbench,
  title={A thin synonymous channel: leakage and probe choices inflate reported gains of codon language models},
  author={Liang, Yuanqing and Zhu, Weimin and Liang, Huiying and Pan, Xiaoyong},
  journal={Nature Communications},
  year={2026},
  note={Preprint: bioRxiv}
}
```