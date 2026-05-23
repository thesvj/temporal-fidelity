#!/usr/bin/env bash
# Run E5-CTRL shuffled-label control probe — ALL models from the gap table.
#
# Covers every model in the paper's representation-behavior gap analysis:
#   molmo2, videochat-flash, qwen2.5-vl, internvl2.5, llava-next-video (both stages)
#   video-llama2 (llm only — Q-Former blocks encoder access)
#   videollama3 (both stages)
#
# Usage:
#   ./run_shuffled_probe.sh                    # all 7 models
#   ./run_shuffled_probe.sh molmo2             # single model
#   LIMIT=20 ./run_shuffled_probe.sh           # quick test run

set -euo pipefail

# --- Configuration -----------------------------------------------------------

declare -A MODEL_STAGE=(
    ["molmo2"]="both"
    ["videochat-flash"]="both"
    ["qwen2.5-vl"]="both"
    ["internvl2.5"]="both"
    ["llava-next-video"]="both"
    ["video-llama2"]="llm"
    ["videollama3"]="both"
)

# Per-model transformers version overrides (must match run.sh)
declare -A TF_PIN=(
    ["internvl2.5"]="transformers>=4.45,<4.46"
    ["videollama3"]="transformers>=5.8"
    ["videochat-flash"]="transformers==4.40.1"
)

N_SHUFFLES=10
LIMIT="${LIMIT:-}"

# --- Parse args: optional model filter ---------------------------------------

if [[ $# -gt 0 ]]; then
    MODELS=("$@")
else
    MODELS=("molmo2" "videochat-flash" "qwen2.5-vl" "internvl2.5" "llava-next-video" "video-llama2" "videollama3")
fi

mkdir -p logs results

# --- Helper -------------------------------------------------------------------

setup_venv() {
    local model="$1"
    local venv=".venv-${model}"
    export UV_PROJECT_ENVIRONMENT="${venv}"
    export UV_LINK_MODE=copy
    uv sync --quiet

    if [[ -n "${TF_PIN[${model}]:-}" ]]; then
        echo "  Pinning ${TF_PIN[${model}]} ..."
        uv pip uninstall transformers --python "${venv}/bin/python" --quiet 2>/dev/null || true
        uv pip install "${TF_PIN[${model}]}" --python "${venv}/bin/python" --quiet
    fi
}

# --- Run ----------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  E5-CTRL: Shuffled-Label Control Probe"
echo "  Models: ${MODELS[*]}"
echo "  $(date)"
echo "================================================================"

FAILED=()

for MODEL in "${MODELS[@]}"; do
    STAGE="${MODEL_STAGE[${MODEL}]:-llm}"
    VENV=".venv-${MODEL}"
    LOG="logs/${MODEL}_shuffled_probe.log"

    # Skip if ALL expected results already exist
    SKIP=1
    if [[ "${STAGE}" == "both" ]]; then
        [[ ! -f "results/${MODEL}_shuffled_probe_llm.csv" ]] && SKIP=0
        [[ ! -f "results/${MODEL}_shuffled_probe_encoder.csv" ]] && SKIP=0
    else
        [[ ! -f "results/${MODEL}_shuffled_probe_${STAGE}.csv" ]] && SKIP=0
    fi

    if [[ "${SKIP}" == "1" ]]; then
        echo ""
        echo "-------- ${MODEL} — SKIPPED (results exist) --------"
        continue
    fi

    echo ""
    echo "-------- ${MODEL} stage=${STAGE} ($(date '+%H:%M:%S')) --------"

    (
        setup_venv "${MODEL}"
        LIMIT_FLAG=""
        [[ -n "${LIMIT}" ]] && LIMIT_FLAG="--limit ${LIMIT}"

        PYTHONUNBUFFERED=1 "${VENV}/bin/python" shuffled_probe.py \
            --models "${MODEL}" \
            --stage "${STAGE}" \
            --n-shuffles "${N_SHUFFLES}" \
            ${LIMIT_FLAG}

    ) 2>&1 | tee "${LOG}" \
        && echo "  OK  -> ${LOG}" \
        || { echo "  FAIL -> ${LOG}"; FAILED+=("${MODEL}"); }
done

# --- Summary ------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  Shuffled probe complete ($(date '+%H:%M:%S'))"
echo "================================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    STAGE="${MODEL_STAGE[${MODEL}]:-llm}"
    if [[ "${STAGE}" == "both" ]]; then
        for s in "llm" "encoder"; do
            FILE="results/${MODEL}_shuffled_probe_${s}.csv"
            if [[ -f "${FILE}" ]]; then
                echo "  ✓ ${FILE} ($(wc -l < "${FILE}") rows)"
            else
                echo "  ✗ ${FILE} — MISSING"
            fi
        done
    else
        FILE="results/${MODEL}_shuffled_probe_${STAGE}.csv"
        if [[ -f "${FILE}" ]]; then
            echo "  ✓ ${FILE} ($(wc -l < "${FILE}") rows)"
        else
            echo "  ✗ ${FILE} — MISSING"
        fi
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "  ⚠ Failed models: ${FAILED[*]}"
    echo "  Check logs/ for details. Re-run with: ./run_shuffled_probe.sh ${FAILED[*]}"
    exit 1
fi
