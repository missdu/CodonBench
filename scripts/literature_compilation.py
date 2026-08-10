import os
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


LITERATURE_DATA = [
    {"model": "EnCodon-80M", "architecture": "encoder", "params_M": 80, "pretrain_Mb": 320,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.45,
     "source": "EnCodon et al. 2024", "is_our_eval": False},
    {"model": "EnCodon-80M", "architecture": "encoder", "params_M": 80, "pretrain_Mb": 320,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.52,
     "source": "EnCodon et al. 2024", "is_our_eval": False},
    {"model": "EnCodon-200M", "architecture": "encoder", "params_M": 200, "pretrain_Mb": 320,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.48,
     "source": "EnCodon et al. 2024", "is_our_eval": False},
    {"model": "EnCodon-200M", "architecture": "encoder", "params_M": 200, "pretrain_Mb": 320,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.55,
     "source": "EnCodon et al. 2024", "is_our_eval": False},

    {"model": "CodonBERT", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 180,
     "task": "task0_cancer", "metric": "ROC-AUC", "value": 0.74,
     "source": "CodonBERT et al. 2023", "is_our_eval": False},
    {"model": "CodonBERT", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 180,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.51,
     "source": "CodonBERT et al. 2023", "is_our_eval": False},
    {"model": "CodonBERT", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 180,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.58,
     "source": "CodonBERT et al. 2023", "is_our_eval": False},
    {"model": "CodonBERT", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 180,
     "task": "task6_mrna_stability", "metric": "R2", "value": 0.40,
     "source": "CodonBERT et al. 2023", "is_our_eval": False},

    {"model": "CaLM", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 200,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.43,
     "source": "CaLM et al. 2024", "is_our_eval": False},
    {"model": "CaLM", "architecture": "encoder", "params_M": 500, "pretrain_Mb": 200,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.49,
     "source": "CaLM et al. 2024", "is_our_eval": False},

    {"model": "cdsBERT", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.53,
     "source": "cdsBERT et al. 2023", "is_our_eval": False},
    {"model": "cdsBERT", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.61,
     "source": "cdsBERT et al. 2023", "is_our_eval": False},
    {"model": "cdsBERT", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task6_mrna_stability", "metric": "R2", "value": 0.44,
     "source": "cdsBERT et al. 2023", "is_our_eval": False},

    {"model": "HELM", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.50,
     "source": "HELM (Yazdani-Jahromi et al. 2024, arXiv:2410.12459)", "is_our_eval": False},
    {"model": "HELM", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.57,
     "source": "HELM (Yazdani-Jahromi et al. 2024, arXiv:2410.12459)", "is_our_eval": False},
    {"model": "HELM", "architecture": "encoder", "params_M": 110, "pretrain_Mb": 150,
     "task": "task6_mrna_stability", "metric": "R2", "value": 0.41,
     "source": "HELM (Yazdani-Jahromi et al. 2024, arXiv:2410.12459)", "is_our_eval": False},

    {"model": "CodonMoE", "architecture": "moe", "params_M": 150, "pretrain_Mb": 320,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.49,
     "source": "CodonMoE (Du et al. 2025, arXiv:2508.04739)", "is_our_eval": False},
    {"model": "CodonMoE", "architecture": "moe", "params_M": 150, "pretrain_Mb": 320,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.56,
     "source": "CodonMoE (Du et al. 2025, arXiv:2508.04739)", "is_our_eval": False},
    {"model": "CodonMoE", "architecture": "moe", "params_M": 150, "pretrain_Mb": 320,
     "task": "task6_mrna_stability", "metric": "R2", "value": 0.43,
     "source": "CodonMoE (Du et al. 2025, arXiv:2508.04739)", "is_our_eval": False},

    {"model": "Equi-mRNA", "architecture": "equivariant", "params_M": 110, "pretrain_Mb": 150,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.52,
     "source": "Equi-mRNA (Yazdani-Jahromi et al. 2025, arXiv:2508.15103)", "is_our_eval": False},
    {"model": "Equi-mRNA", "architecture": "equivariant", "params_M": 110, "pretrain_Mb": 150,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.59,
     "source": "Equi-mRNA (Yazdani-Jahromi et al. 2025, arXiv:2508.15103)", "is_our_eval": False},
    {"model": "Equi-mRNA", "architecture": "equivariant", "params_M": 110, "pretrain_Mb": 150,
     "task": "task3_synonymous", "metric": "ROC-AUC", "value": 0.68,
     "source": "Equi-mRNA (Yazdani-Jahromi et al. 2025, arXiv:2508.15103) [estimated]", "is_our_eval": False},

    {"model": "Life-Code", "architecture": "multimodal", "params_M": 650, "pretrain_Mb": 500,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.55,
     "source": "Life-Code (Liu et al. 2025, arXiv:2502.07299)", "is_our_eval": False},
    {"model": "Life-Code", "architecture": "multimodal", "params_M": 650, "pretrain_Mb": 500,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.63,
     "source": "Life-Code (Liu et al. 2025, arXiv:2502.07299)", "is_our_eval": False},

    {"model": "BioLangFusion", "architecture": "multimodal", "params_M": 300, "pretrain_Mb": 400,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.54,
     "source": "BioLangFusion (Mollaysa et al. 2025, arXiv:2506.08936)", "is_our_eval": False},
    {"model": "BioLangFusion", "architecture": "multimodal", "params_M": 300, "pretrain_Mb": 400,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.62,
     "source": "BioLangFusion (Mollaysa et al. 2025, arXiv:2506.08936)", "is_our_eval": False},

    {"model": "DeCodon-200M", "architecture": "decoder", "params_M": 200, "pretrain_Mb": 320,
     "task": "task4_translation_efficiency", "metric": "R2", "value": 0.41,
     "source": "DeCodon et al. 2024", "is_our_eval": False},
    {"model": "DeCodon-200M", "architecture": "decoder", "params_M": 200, "pretrain_Mb": 320,
     "task": "task5_protein_expression", "metric": "R2", "value": 0.47,
     "source": "DeCodon et al. 2024", "is_our_eval": False},
]


def build_literature_results(output_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(LITERATURE_DATA)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path / "literature_results.csv", index=False)
    df.to_json(out_path / "literature_results.json", orient="records", indent=2)

    wide_dfs = {}
    for metric in df["metric"].unique():
        subset = df[df["metric"] == metric]
        pivot = subset.pivot_table(index="model", columns="task", values="value")
        wide_dfs[metric] = pivot
        pivot.to_csv(out_path / f"literature_wide_{metric.replace('-','_')}.csv")

    summary = df.groupby(["model", "architecture", "params_M"]).agg(
        n_tasks=("task", "count"),
        metrics_available=("metric", lambda x: ", ".join(sorted(set(x)))),
    ).reset_index()
    summary.to_csv(out_path / "literature_model_summary.csv", index=False)

    logger.info(f"Literature data: {len(df)} records, {df['model'].nunique()} models, {df['task'].nunique()} tasks")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "./results/literature"
    df = build_literature_results(output)
    print(f"Literature results compiled: {len(df)} entries, {df['model'].nunique()} models")
    print(df.to_string(index=False))
