#!/usr/bin/env bash
# Run all experiments, one model at a time in isolated venvs.
#
# Usage:
#   ./run.sh                                          # all models
#   ./run.sh llava-next-video qwen2.5-vl              # specific models
#   LIMIT=50 ./run.sh                                 # cap videos per interval
#   SKIP_LAYERS=1 ./run.sh                            # skip E005

set -euo pipefail

ALL_MODELS=("llava-next-video" "video-llama2" "qwen2.5-vl" "molmo2" "internvl2.5" "videollama3" "videochat-flash")
MODELS=("${@:-${ALL_MODELS[@]}}")
LIMIT="${LIMIT:-}"
SKIP_LAYERS="${SKIP_LAYERS:-0}"

mkdir -p logs results figs

# Per-model transformers version overrides (most models use 4.57.x from lockfile)
declare -A TF_PIN=(
    ["internvl2.5"]="transformers>=4.45,<4.46"
    ["videollama3"]="transformers>=5.8"
    ["videochat-flash"]="transformers==4.40.1"
)

for MODEL in "${MODELS[@]}"; do
    VENV=".venv-${MODEL}"
    LOG="logs/${MODEL}.log"
    echo ""
    echo "======== ${MODEL} ($(date '+%H:%M:%S')) ========"

    (
        export UV_PROJECT_ENVIRONMENT="${VENV}"
        export UV_LINK_MODE=copy
        uv sync --quiet

        # Pin transformers if this model needs a specific version
        if [[ -n "${TF_PIN[${MODEL}]:-}" ]]; then
            echo "  Pinning ${TF_PIN[${MODEL}]} ..."
            uv pip uninstall transformers --python "${VENV}/bin/python" --quiet 2>/dev/null || true
            uv pip install "${TF_PIN[${MODEL}]}" --python "${VENV}/bin/python" --quiet
        fi

        EXTRA=""
        [[ "${SKIP_LAYERS}" == "1" ]] && EXTRA+=" --skip-layers"
        [[ -n "${LIMIT}" ]] && EXTRA+=" --limit ${LIMIT}"

        "${VENV}/bin/python" run_all.py \
            --models "${MODEL}" \
            --skip-gen \
            --skip-analyze \
            ${EXTRA}

    ) 2>&1 | tee "${LOG}" \
        && echo "  OK  -> logs/${MODEL}.log" \
        || echo "  FAIL -> logs/${MODEL}.log"
done

# Final analysis (use first available venv)
echo ""
echo "======== Analysis ========"
FIRST_VENV=".venv-${MODELS[0]}"
UV_PROJECT_ENVIRONMENT="${FIRST_VENV}" UV_LINK_MODE=copy \
    uv run python analyze.py 2>&1 | tee logs/analyze.log

echo ""
echo "Done.  Results -> results/  Figures -> figs/"
