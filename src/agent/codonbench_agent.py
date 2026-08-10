"""
CodonBench-Agent: AI Agent Framework for Automated Benchmark Evaluation
======================================================================
Three agents:
  - ExecutionAgent: 自动化评估执行 + 评估artifact pre-flight检测
  - RecommendationAgent: 智能模型推荐
  - MaintenanceAgent: 持续监控新论文/新模型
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when gene-identity leakage is detected in train/test splits."""
    pass


class SelectionBiasError(Exception):
    """Raised when best-epoch selection uses test-set performance."""
    pass


def check_gene_leakage(train_ids, test_ids, gene_map) -> Dict:
    """Pre-flight check: flag if any gene appears in both train and test.

    Parameters
    ----------
    train_ids : array-like
        Sample indices or identifiers in the training set.
    test_ids : array-like
        Sample indices or identifiers in the test set.
    gene_map : dict or array-like
        Mapping from sample id to gene id.

    Returns
    -------
    dict with keys: detected, n_overlapping_genes, n_train_genes,
                    n_test_genes, overlap_ratio, example_genes

    Raises
    ------
    LeakageError
        If any gene appears in both train and test sets.
    """
    if gene_map is None:
        return {"detected": None, "reason": "gene_map not provided; check skipped"}

    train_genes = {gene_map[i] for i in train_ids if i in gene_map}
    test_genes = {gene_map[i] for i in test_ids if i in gene_map}
    overlap = train_genes & test_genes
    n_overlap = len(overlap)
    n_test = len(test_genes)
    ratio = n_overlap / n_test if n_test > 0 else 0.0

    result = {
        "detected": n_overlap > 0,
        "n_overlapping_genes": n_overlap,
        "n_train_genes": len(train_genes),
        "n_test_genes": n_test,
        "overlap_ratio": round(ratio, 4),
        "example_genes": sorted(overlap)[:5],
    }

    if n_overlap > 0:
        raise LeakageError(
            f"Gene-identity leakage detected: {n_overlap} genes appear in both "
            f"train and test sets (overlap ratio = {ratio:.2%}). "
            f"Use LOGO-CV splits (--split logo) to prevent this artifact."
        )

    return result


def check_epoch_selection(selection_metric_source: str) -> Dict:
    """Pre-flight check: flag if probe epoch is chosen on the test set.

    Parameters
    ----------
    selection_metric_source : str
        One of 'test', 'validation', 'val', 'train', or 'unknown'.

    Returns
    -------
    dict with keys: detected, selection_source

    Raises
    ------
    SelectionBiasError
        If epoch selection uses the test set.
    """
    source_lower = selection_metric_source.lower().strip()
    uses_test = source_lower in ("test", "test_set", "testing")

    result = {
        "detected": uses_test,
        "selection_source": selection_metric_source,
    }

    if uses_test:
        raise SelectionBiasError(
            "Best-epoch selection bias detected: probe epoch is selected on "
            "the test set, manufacturing an optimistic nonlinear advantage. "
            "Use validation-based early stopping (--early-stop val)."
        )

    return result


@dataclass
class ExecutionLog:
    model_name: str
    task_name: str
    start_time: str = ""
    end_time: str = ""
    status: str = "pending"
    device: str = "cuda:0"
    batch_size: int = 8
    random_seed: int = 42
    quantized: bool = False
    vram_peak_mb: float = 0
    error: Optional[str] = None
    result: Optional[Dict] = None
    artifact_warnings: List[Dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class ExecutionAgent:
    def __init__(self, project_root: str, device: str = "cuda:0"):
        self.project_root = Path(project_root)
        self.device = device
        self.logs: List[ExecutionLog] = []
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def plan_evaluation(self, model_names: List[str], task_names: List[str]) -> List[Dict]:
        configs_path = self.project_root / "configs" / "models.yaml"
        import yaml
        with open(configs_path, "r") as f:
            model_configs = yaml.safe_load(f).get("models", {})

        plan = []
        for model_name in model_names:
            mcfg = model_configs.get(model_name, {})
            params_m = mcfg.get("params_M", 100)
            plan.append({
                "model": model_name,
                "tasks": task_names,
                "params_M": params_m,
                "estimated_vram_gb": params_m * 0.004 + 0.5,
                "priority": 1 if params_m <= 200 else 2,
            })
        plan.sort(key=lambda x: x["params_M"])
        return plan

    def preflight_check(
        self,
        train_ids=None,
        test_ids=None,
        gene_map=None,
        epoch_selection_source: str = "unknown",
    ) -> List[Dict]:
        """Run pre-flight methodological validation checks.

        Checks for two evaluation artifacts identified by this study:
          1. Gene-identity leakage (variants from same gene in train & test)
          2. Best-epoch selection bias (epoch chosen on test set)

        Returns list of warning dicts; raises LeakageError/SelectionBiasError
        if a critical artifact is detected.
        """
        warnings = []

        if train_ids is not None and test_ids is not None and gene_map is not None:
            try:
                check_gene_leakage(train_ids, test_ids, gene_map)
            except LeakageError as e:
                warnings.append({
                    "artifact_type": "gene_identity_leakage",
                    "severity": "critical",
                    "message": str(e),
                })
                raise

        if epoch_selection_source != "unknown":
            try:
                check_epoch_selection(epoch_selection_source)
            except SelectionBiasError as e:
                warnings.append({
                    "artifact_type": "best_epoch_selection_bias",
                    "severity": "critical",
                    "message": str(e),
                })
                raise

        return warnings

    def execute_with_recovery(
        self,
        model_name: str,
        task_name: str,
        task_data: pd.DataFrame,
        eval_func,
        max_retries: int = 3,
        train_ids=None,
        test_ids=None,
        gene_map=None,
        epoch_selection_source: str = "unknown",
    ) -> ExecutionLog:
        log = ExecutionLog(
            model_name=model_name,
            task_name=task_name,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            device=self.device,
        )

        try:
            self.preflight_check(
                train_ids=train_ids,
                test_ids=test_ids,
                gene_map=gene_map,
                epoch_selection_source=epoch_selection_source,
            )
        except (LeakageError, SelectionBiasError) as e:
            log.status = "blocked"
            log.error = str(e)
            log.artifact_warnings = [{
                "artifact_type": (
                    "gene_identity_leakage"
                    if isinstance(e, LeakageError)
                    else "best_epoch_selection_bias"
                ),
                "severity": "critical",
                "message": str(e),
            }]
            log.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.logs.append(log)
            self._save_log(log)
            logger.warning(f"Pre-flight check blocked evaluation: {e}")
            return log

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[Attempt {attempt}] Evaluating {model_name} on {task_name}")
                result = eval_func(model_name, task_name, task_data, device=self.device)
                log.status = "success"
                log.result = result
                log.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self.logs.append(log)
                self._save_log(log)
                return log

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on attempt {attempt}, trying recovery...")
                    import torch
                    torch.cuda.empty_cache()

                    if not log.quantized and attempt < max_retries:
                        log.quantized = True
                        logger.info("Retrying with 4-bit quantization")
                        continue

                    if attempt < max_retries:
                        log.batch_size = max(1, log.batch_size // 2)
                        logger.info(f"Retrying with batch_size={log.batch_size}")
                        continue

                log.status = "failed"
                log.error = str(e)
                break

            except Exception as e:
                log.status = "failed"
                log.error = str(e)
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                break

        log.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(log)
        self._save_log(log)
        return log

    def _save_log(self, log: ExecutionLog):
        log_dir = self.results_dir / "execution_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{log.model_name}_{log.task_name}.json"
        with open(log_path, "w") as f:
            f.write(log.to_json())

    def get_summary(self) -> pd.DataFrame:
        if not self.logs:
            return pd.DataFrame()
        records = []
        for log in self.logs:
            records.append({
                "model": log.model_name,
                "task": log.task_name,
                "status": log.status,
                "quantized": log.quantized,
                "batch_size": log.batch_size,
                "error": log.error,
                "artifact_warnings": log.artifact_warnings,
                "start": log.start_time,
                "end": log.end_time,
            })
        return pd.DataFrame(records)


class RecommendationAgent:
    def __init__(self, results_db: pd.DataFrame, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1):
        self.results_db = results_db
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.OPEN_SOURCE_MODELS = {
            "encodon-80m", "encodon-200m", "decodon-200m", "calm", "codonbert",
            "cdsbert", "helm", "codonmoe", "equi_mrna",
        }

    def parse_query(self, query: str) -> Dict:
        query_lower = query.lower()
        requirements = {
            "task_type": None,
            "max_params_m": None,
            "require_open_source": False,
            "architecture_preference": None,
        }

        task_keywords = {
            "task0": "task0_cancer", "cancer": "task0_cancer", "hotspot": "task0_cancer",
            "task3": "task3_synonymous", "synonymous": "task3_synonymous",
            "task4": "task4_translation_efficiency", "translation": "task4_translation_efficiency",
            "task5": "task5_protein_expression", "expression": "task5_protein_expression", "protein expr": "task5_protein_expression",
            "task6": "task6_mrna_stability", "stability": "task6_mrna_stability", "mrna": "task6_mrna_stability",
        }
        for kw, task in task_keywords.items():
            if kw in query_lower:
                requirements["task_type"] = task
                break

        arch_keywords = {"encoder": "encoder", "decoder": "decoder", "moe": "moe", "equivariant": "equivariant", "multimodal": "multimodal"}
        for kw, arch in arch_keywords.items():
            if kw in query_lower:
                requirements["architecture_preference"] = arch
                break

        if any(kw in query_lower for kw in ["open", "opensource", "open-source", "free"]):
            requirements["require_open_source"] = True

        if any(kw in query_lower for kw in ["small", "lightweight", "light", "efficient"]):
            requirements["max_params_m"] = 200

        if any(kw in query_lower for kw in ["best", "overall", "top", "highest"]):
            requirements["task_type"] = "overall"

        return requirements

    def score_models(self, requirements: Dict) -> pd.DataFrame:
        models = self.results_db["model"].unique()
        scores = {}

        for model in models:
            model_data = self.results_db[self.results_db["model"] == model]

            if requirements["task_type"] and requirements["task_type"] != "overall":
                task_data = model_data[model_data["task"] == requirements["task_type"]]
                perf = task_data["value"].mean() if len(task_data) > 0 else 0
            else:
                perf = model_data["value"].mean()

            params_m = model_data["params_M"].iloc[0] if "params_M" in model_data.columns else 100
            cost = params_m / 1000.0

            open_source = 1.0 if model in self.OPEN_SOURCE_MODELS else 0.0

            score = self.alpha * perf + self.beta * (1.0 / (1.0 + cost)) + self.gamma * open_source

            if requirements["max_params_m"] and params_m > requirements["max_params_m"]:
                score *= 0.5

            if requirements["architecture_preference"]:
                arch = model_data["architecture"].iloc[0] if "architecture" in model_data.columns else ""
                if arch != requirements["architecture_preference"]:
                    score *= 0.7

            if requirements["require_open_source"] and model not in self.OPEN_SOURCE_MODELS:
                score *= 0.3

            scores[model] = score

        result = pd.DataFrame([
            {"model": m, "score": s} for m, s in scores.items()
        ]).sort_values("score", ascending=False)
        return result

    def recommend(self, query: str, top_k: int = 3) -> Dict:
        requirements = self.parse_query(query)
        ranked = self.score_models(requirements)
        top = ranked.head(top_k)

        recommendations = []
        for _, row in top.iterrows():
            model = row["model"]
            model_data = self.results_db[self.results_db["model"] == model]
            arch = model_data["architecture"].iloc[0] if "architecture" in model_data.columns else "unknown"
            params = model_data["params_M"].iloc[0] if "params_M" in model_data.columns else 0

            code_example = f"""from src.models.loader import CodonModelLoader
model, tokenizer, meta = CodonModelLoader.load("{model}", device="cuda:0")
# Architecture: {arch}, Parameters: {params}M
# Score: {row['score']:.4f}"""

            recommendations.append({
                "model": model,
                "score": round(row["score"], 4),
                "architecture": arch,
                "params_M": int(params),
                "code_example": code_example,
            })

        return {
            "query": query,
            "parsed_requirements": requirements,
            "recommendations": recommendations,
        }


class MaintenanceAgent:
    SEARCH_KEYWORDS = [
        "codon language model",
        "mRNA language model",
        "codonBERT",
        "codon optimization model",
        "synonymous codon model",
    ]

    def __init__(self, results_dir: str, leaderboard_path: str = None):
        self.results_dir = Path(results_dir)
        self.leaderboard_path = Path(leaderboard_path) if leaderboard_path else self.results_dir / "leaderboard.json"
        self.known_models = set()
        self._load_leaderboard()

    def _load_leaderboard(self):
        if self.leaderboard_path.exists():
            with open(self.leaderboard_path, "r") as f:
                data = json.load(f)
            self.known_models = set(data.get("models", []))
            logger.info(f"Loaded leaderboard: {len(self.known_models)} known models")

    def search_arxiv(self, keywords: List[str] = None) -> List[Dict]:
        if keywords is None:
            keywords = self.SEARCH_KEYWORDS

        papers = []
        try:
            import requests
            for kw in keywords:
                url = f"http://export.arxiv.org/api/query?search_query=all:{kw}&max_results=5&sortBy=submittedDate&sortOrder=descending"
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(resp.text)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall("atom:entry", ns):
                        title = entry.find("atom:title", ns).text.strip()
                        arxiv_id = entry.find("atom:id", ns).text.strip()
                        published = entry.find("atom:published", ns).text.strip()
                        summary = entry.find("atom:summary", ns).text.strip()[:200]
                        papers.append({
                            "title": title,
                            "arxiv_id": arxiv_id,
                            "published": published,
                            "summary": summary,
                            "keyword": kw,
                        })
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

        return papers

    def detect_new_models(self, papers: List[Dict]) -> List[Dict]:
        new_models = []
        model_name_patterns = {
            "encodon": "EnCodon", "decodon": "DeCodon", "codonbert": "CodonBERT",
            "calm": "CaLM", "cdsbert": "cdsBERT", "helm": "HELM",
            "codonmoe": "CodonMoE", "equi-mrna": "Equi-mRNA", "equimrna": "Equi-mRNA",
            "life-code": "Life-Code", "lifecode": "Life-Code",
            "biolangfusion": "BioLangFusion",
            "codongpt": "CodonGPT", "synodonlm": "SynCodonLM",
        }

        for paper in papers:
            text = (paper["title"] + " " + paper["summary"]).lower()
            for pattern, model_name in model_name_patterns.items():
                if pattern in text and model_name not in self.known_models:
                    new_models.append({
                        "model_name": model_name,
                        "source_paper": paper["arxiv_id"],
                        "paper_title": paper["title"],
                        "published": paper["published"],
                    })

        return new_models

    def update_leaderboard(self, new_models: List[Dict]):
        if not new_models:
            return

        data = {"models": list(self.known_models), "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}
        for nm in new_models:
            data["models"].append(nm["model_name"])
            self.known_models.add(nm["model_name"])
            logger.info(f"New model detected: {nm['model_name']} from {nm['source_paper']}")

        with open(self.leaderboard_path, "w") as f:
            json.dump(data, f, indent=2)

    def run_check(self) -> Dict:
        logger.info("Running maintenance check...")
        papers = self.search_arxiv()
        new_models = self.detect_new_models(papers)
        self.update_leaderboard(new_models)

        return {
            "papers_found": len(papers),
            "new_models_detected": len(new_models),
            "new_models": new_models,
            "total_known_models": len(self.known_models),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    project_root = Path(__file__).resolve().parents[2]

    print("=" * 60)
    print("CodonBench-Agent Demo")
    print("=" * 60)

    print("\n--- Pre-flight Artifact Detection Demo ---")
    from src.agent.codonbench_agent import check_gene_leakage, check_epoch_selection, LeakageError, SelectionBiasError

    gene_map = {0: "BRCA1", 1: "BRCA1", 2: "TP53", 3: "EGFR", 4: "EGFR", 5: "KRAS"}
    train_ids = [0, 1, 2]
    test_ids = [3, 4, 5]
    try:
        check_gene_leakage(train_ids, test_ids, gene_map)
        print("  No leakage detected (disjoint genes)")
    except LeakageError as e:
        print(f"  LEAKAGE DETECTED: {e}")

    train_ids_safe = [0, 1, 2]
    test_ids_safe = [3, 4, 5]
    gene_map_safe = {0: "BRCA1", 1: "BRCA1", 2: "TP53", 3: "EGFR", 4: "KRAS", 5: "BRAF"}
    try:
        result = check_gene_leakage(train_ids_safe, test_ids_safe, gene_map_safe)
        print(f"  No leakage detected: {result}")
    except LeakageError as e:
        print(f"  LEAKAGE DETECTED: {e}")

    try:
        check_epoch_selection("test")
        print("  No selection bias detected")
    except SelectionBiasError as e:
        print(f"  SELECTION BIAS DETECTED: {e}")

    try:
        result = check_epoch_selection("validation")
        print(f"  No selection bias detected: {result}")
    except SelectionBiasError as e:
        print(f"  SELECTION BIAS DETECTED: {e}")

    print("\n--- Recommendation Agent ---")
    from scripts.literature_compilation import build_literature_results
    df = build_literature_results(str(project_root / "results" / "literature"))

    rec_agent = RecommendationAgent(df)
    queries = [
        "Best model for synonymous variant prediction",
        "Lightweight model for translation efficiency",
        "Best overall model with open source weights",
    ]
    for q in queries:
        result = rec_agent.recommend(q)
        print(f"\nQuery: {q}")
        for rec in result["recommendations"]:
            print(f"  #{result['recommendations'].index(rec)+1}  {rec['model']:20s}  score={rec['score']:.4f}  {rec['architecture']:12s}  {rec['params_M']}M")

    print("\n--- Maintenance Agent ---")
    maint_agent = MaintenanceAgent(str(project_root / "results"))
    check = maint_agent.run_check()
    print(f"  Papers found: {check['papers_found']}")
    print(f"  New models: {check['new_models_detected']}")
    print(f"  Total known: {check['total_known_models']}")
