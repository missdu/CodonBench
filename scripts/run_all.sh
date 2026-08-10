#!/bin/bash
# ==============================================================================
# CodonBench: One-Click Full Evaluation Pipeline
# ==============================================================================
# 用法:
#   bash run_all.sh              # 完整流程（setup + smoke + eval）
#   bash run_all.sh --skip-setup # 跳过环境安装，直接评估
#   bash run_all.sh --smoke-only # 只跑冒烟测试，不跑完整评估
# ==============================================================================

set -euo pipefail

SKIP_SETUP=false
SMOKE_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-setup)  SKIP_SETUP=true ;;
        --smoke-only)  SMOKE_ONLY=true ;;
    esac
done

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${PROJECT_ROOT}/data"
RESULTS_DIR="${PROJECT_ROOT}/results"
FIGURES_DIR="${PROJECT_ROOT}/figures"
LOGS_DIR="${PROJECT_ROOT}/logs"

CONDA_BASE="<HOME>/anaconda3"
ENV_PATH="${PROJECT_ROOT}/conda_env"
PYTHON="${CONDA_BASE}/bin/conda run -p ${ENV_PATH} python"

# GPU allocation
ZS_DEVICE="cuda:0"
DS_DEVICE="cuda:2"

# Models
ZS_MODELS="encodon-80m,encodon-200m,decodon-200m,calm,codonbert,cdsbert"
DS_MODELS="encodon-80m,encodon-200m,decodon-200m,calm,codonbert,cdsbert"

# Tasks
ZS_TASKS="task0_cancer,task3_synonymous"
DS_TASKS="task4_translation_efficiency,task5_protein_expression"

# ── Create directories ─────────────────────────────────────────────────────────
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}/zeroshot" "${RESULTS_DIR}/downstream" \
         "${RESULTS_DIR}/summary" "${FIGURES_DIR}" "${LOGS_DIR}"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE="${LOGS_DIR}/run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "CodonBench Full Evaluation Pipeline"
echo "Started: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "=============================================================================="

# ── Step 0: Environment Setup ──────────────────────────────────────────────────
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo "[Step 0/6] Setting up environment..."
    if [ ! -d "${ENV_PATH}" ]; then
        echo "  Creating conda env at ${ENV_PATH}..."
        ${CONDA_BASE}/bin/conda create -p "${ENV_PATH}" python=3.11 -y
        $PYTHON -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
        $PYTHON -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        $PYTHON -m pip install \
            transformers datasets scikit-learn scipy biopython pandas numpy \
            matplotlib seaborn pyyaml tqdm requests statsmodels \
            bitsandbytes huggingface-hub safetensors accelerate
        echo "  Environment created."
    else
        echo "  Environment already exists at ${ENV_PATH}."
    fi
else
    echo "[Step 0/6] Skipping setup (--skip-setup)."
fi

# ── Step 1: Smoke Test ────────────────────────────────────────────────────────
echo ""
echo "[Step 1/6] Running smoke tests (小样本验证)..."
SMOKE_RESULT=0
$PYTHON "${SCRIPT_DIR}/smoke_test.py" --output "${RESULTS_DIR}/smoke_test" || SMOKE_RESULT=$?
if [ $SMOKE_RESULT -ne 0 ]; then
    echo "WARNING: Smoke tests had failures. Review ${RESULTS_DIR}/smoke_test/smoke_test_report.json"
    echo "Critical failures may prevent evaluation. Non-critical ones (e.g., HuggingFace download) are OK."
else
    echo "All smoke tests passed!"
fi

if [ "$SMOKE_ONLY" = true ]; then
    echo "Smoke-only mode. Exiting."
    exit $SMOKE_RESULT
fi

# ── Step 2: Data Preparation ───────────────────────────────────────────────────
echo ""
echo "[Step 2/6] Preparing datasets..."
$PYTHON -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.data.download import prepare_all_tasks
results = prepare_all_tasks('${DATA_DIR}')
for k, v in results.items():
    status = f'OK ({len(v)} samples)' if v is not None else 'FAILED'
    print(f'  {k}: {status}')
" || echo "WARNING: Data preparation had issues, will use fallback data."

# ── Step 3: Zero-shot Evaluation ───────────────────────────────────────────────
echo ""
echo "[Step 3/6] Running zero-shot mutation prediction evaluation..."
echo "  Models: ${ZS_MODELS}"
echo "  Tasks:  ${ZS_TASKS}"
echo "  Device: ${ZS_DEVICE}"
$PYTHON "${SCRIPT_DIR}/run_zeroshot.py" \
    --data "${DATA_DIR}" \
    --output "${RESULTS_DIR}/zeroshot" \
    --device "${ZS_DEVICE}" \
    --models "${ZS_MODELS}" \
    --tasks "${ZS_TASKS}" \
    || echo "WARNING: Some zero-shot evaluations failed."

# ── Step 4: Downstream Evaluation ──────────────────────────────────────────────
echo ""
echo "[Step 4/6] Running downstream property prediction evaluation..."
echo "  Models: ${DS_MODELS}"
echo "  Tasks:  ${DS_TASKS}"
echo "  Device: ${DS_DEVICE}"
$PYTHON "${SCRIPT_DIR}/run_downstream.py" \
    --data "${DATA_DIR}" \
    --output "${RESULTS_DIR}/downstream" \
    --device "${DS_DEVICE}" \
    --models "${DS_MODELS}" \
    --tasks "${DS_TASKS}" \
    || echo "WARNING: Some downstream evaluations failed."

# ── Step 5: Statistical Analysis ───────────────────────────────────────────────
echo ""
echo "[Step 5/6] Running statistical analysis..."
$PYTHON "${SCRIPT_DIR}/run_analysis.py" \
    --results "${RESULTS_DIR}" \
    --output "${RESULTS_DIR}/summary" \
    || echo "WARNING: Analysis had issues."

# ── Step 6: Figure Generation ─────────────────────────────────────────────────
echo ""
echo "[Step 6/6] Generating publication-quality figures..."
$PYTHON "${SCRIPT_DIR}/run_figures.py" \
    --results "${RESULTS_DIR}" \
    --output "${FIGURES_DIR}" \
    || echo "WARNING: Figure generation had issues."

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================================================="
echo "CodonBench Evaluation COMPLETE"
echo "Finished: $(date)"
echo ""
echo "Results:"
echo "  Smoke test:  ${RESULTS_DIR}/smoke_test/smoke_test_report.json"
echo "  Zero-shot:   ${RESULTS_DIR}/zeroshot/"
echo "  Downstream:   ${RESULTS_DIR}/downstream/"
echo "  Summary:      ${RESULTS_DIR}/summary/"
echo "  Figures:      ${FIGURES_DIR}/"
echo "  Log:          ${LOG_FILE}"
echo ""
ls -la "${FIGURES_DIR}"/*.png 2>/dev/null || echo "  (no PNG figures)"
echo "=============================================================================="
