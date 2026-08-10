import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_all_results(results_dir: str) -> pd.DataFrame:
    results = []
    rpath = Path(results_dir)
    for json_file in rpath.rglob("*.json"):
        if json_file.name in ("zeroshot_summary.json", "downstream_summary.json", "all_results_summary.json"):
            continue
        try:
            with open(json_file, "r") as f:
                r = json.load(f)
            if isinstance(r, dict) and "model" in r:
                results.append(r)
        except Exception:
            continue

    for summary_name in ("zeroshot_summary.json", "downstream_summary.json"):
        sp = rpath / summary_name
        if sp.exists():
            with open(sp, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                results.extend(data)

    if not results:
        logger.warning("No results found!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[df.get("success", False) == True].copy()
    return df


def compute_rankings(df: pd.DataFrame, metric_col: str = None) -> pd.DataFrame:
    if df.empty:
        return df

    if metric_col is None:
        if "ROC-AUC" in df.columns:
            metric_col = "ROC-AUC"
        elif "R2" in df.columns:
            metric_col = "R2"
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            metric_col = numeric_cols[0] if numeric_cols else None

    if metric_col is None:
        return df

    df = df.copy()
    df["rank"] = df.groupby("task")[metric_col].rank(ascending=False, method="min")
    df["avg_rank"] = df.groupby("model")["rank"].transform("mean")
    return df.sort_values("avg_rank")


def pairwise_significance_test(df: pd.DataFrame, metric_col: str = "ROC-AUC") -> pd.DataFrame:
    if df.empty or "model" not in df.columns or "task" not in df.columns:
        return pd.DataFrame()

    models = df["model"].unique()
    tasks = df["task"].unique()
    records = []

    for m1, m2 in combinations(models, 2):
        m1_vals = []
        m2_vals = []
        for task in tasks:
            v1 = df[(df["model"] == m1) & (df["task"] == task)][metric_col]
            v2 = df[(df["model"] == m2) & (df["task"] == task)][metric_col]
            if len(v1) > 0 and len(v2) > 0:
                m1_vals.append(v1.values[0])
                m2_vals.append(v2.values[0])

        if len(m1_vals) < 2:
            continue

        try:
            t_stat, p_val = stats.ttest_rel(m1_vals, m2_vals)
            diff = np.mean(m1_vals) - np.mean(m2_vals)
            cohen_d = diff / np.std(np.array(m1_vals) - np.array(m2_vals)) if np.std(np.array(m1_vals) - np.array(m2_vals)) > 0 else 0
        except Exception:
            t_stat, p_val, diff, cohen_d = float("nan"), float("nan"), float("nan"), float("nan")

        records.append({
            "model_1": m1, "model_2": m2,
            "mean_diff": round(diff, 4),
            "t_stat": round(t_stat, 3) if not np.isnan(t_stat) else None,
            "p_value": p_val,
            "cohen_d": round(cohen_d, 3) if not np.isnan(cohen_d) else None,
            "n_tasks": len(m1_vals),
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty and "p_value" in result_df.columns:
        from statsmodels.stats.multitest import multipletests as sm_multipletests
        valid_mask = result_df["p_value"].notna()
        if valid_mask.sum() > 0:
            _, p_corrected, _, _ = sm_multipletests(
                result_df.loc[valid_mask, "p_value"].values, method="fdr_bh"
            )
            result_df.loc[valid_mask, "p_value_fdr"] = p_corrected
            result_df["significant_fdr_005"] = result_df.get("p_value_fdr", 1.0) < 0.05

    return result_df


def architecture_effect_analysis(df: pd.DataFrame, metric_col: str = "ROC-AUC") -> pd.DataFrame:
    if df.empty or "architecture" not in df.columns:
        return pd.DataFrame()

    arch_stats = df.groupby("architecture")[metric_col].agg(["mean", "std", "count"]).reset_index()
    arch_stats.columns = ["architecture", f"mean_{metric_col}", f"std_{metric_col}", "n"]

    architectures = arch_stats["architecture"].unique()
    records = []
    for a1, a2 in combinations(architectures, 2):
        v1 = df[df["architecture"] == a1][metric_col].values
        v2 = df[df["architecture"] == a2][metric_col].values
        if len(v1) > 1 and len(v2) > 1:
            u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            records.append({
                "arch_1": a1, "arch_2": a2,
                "mean_1": round(np.mean(v1), 4), "mean_2": round(np.mean(v2), 4),
                "p_value": p_val,
            })

    return pd.DataFrame(records)


def parameter_scaling_analysis(df: pd.DataFrame, metric_col: str = "ROC-AUC") -> Dict:
    if df.empty or "params_M" not in df.columns:
        return {}

    avg_by_model = df.groupby("model").agg({
        metric_col: "mean", "params_M": "first", "architecture": "first"
    }).reset_index()

    params = avg_by_model["params_M"].values
    perf = avg_by_model[metric_col].values

    if len(params) < 3:
        return {}

    log_params = np.log10(params)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, perf)
    spearman_r, spearman_p = stats.spearmanr(params, perf)

    return {
        "log_linear_slope": round(slope, 4),
        "log_linear_intercept": round(intercept, 4),
        "log_linear_r2": round(r_value**2, 4),
        "log_linear_p": p_value,
        "spearman_r": round(spearman_r, 4),
        "spearman_p": spearman_p,
        "conclusion": "Significant parameter scaling effect" if p_value < 0.05 else "No significant parameter scaling effect",
    }


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05
) -> tuple:
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.nanmean(sample))
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return round(lower, 4), round(upper, 4)


def comprehensive_analysis(results_dir: str, output_dir: str) -> Dict[str, Any]:
    df = load_all_results(results_dir)

    if df.empty:
        logger.error("No results to analyze!")
        return {}

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    analysis = {}

    df_ranked = compute_rankings(df)
    analysis["rankings"] = df_ranked
    df_ranked.to_csv(out_path / "model_rankings.csv", index=False)

    zs_df = df[df["task"].str.contains("task[0-3]")]
    if not zs_df.empty and "ROC-AUC" in zs_df.columns:
        pairwise = pairwise_significance_test(zs_df, "ROC-AUC")
        analysis["pairwise_tests"] = pairwise
        pairwise.to_csv(out_path / "pairwise_tests.csv", index=False)

        arch_effect = architecture_effect_analysis(zs_df, "ROC-AUC")
        analysis["architecture_effect"] = arch_effect
        arch_effect.to_csv(out_path / "architecture_effect.csv", index=False)

        scaling = parameter_scaling_analysis(zs_df, "ROC-AUC")
        analysis["scaling_analysis"] = scaling

    ds_df = df[df["task"].str.contains("task[4-6]")]
    if not ds_df.empty and "R2" in ds_df.columns:
        ds_ranked = compute_rankings(ds_df, "R2")
        analysis["downstream_rankings"] = ds_ranked
        ds_ranked.to_csv(out_path / "downstream_rankings.csv", index=False)

        ds_scaling = parameter_scaling_analysis(ds_df, "R2")
        analysis["downstream_scaling"] = ds_scaling

    summary_path = out_path / "analysis_summary.json"
    serializable = {}
    for k, v in analysis.items():
        if isinstance(v, pd.DataFrame):
            serializable[k] = v.to_dict(orient="records")
        elif isinstance(v, dict):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    all_results_path = out_path / "all_results_summary.json"
    df.to_json(all_results_path, orient="records", indent=2)

    logger.info(f"Analysis complete. Results in {out_path}")
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results directory with JSON files")
    parser.add_argument("--output", required=True, help="Output directory for analysis")
    args = parser.parse_args()
    analysis = comprehensive_analysis(args.results, args.output)
    print("Analysis complete.")
    for k, v in analysis.items():
        if isinstance(v, pd.DataFrame):
            print(f"  {k}: {len(v)} rows")
        elif isinstance(v, dict):
            print(f"  {k}: {v}")
