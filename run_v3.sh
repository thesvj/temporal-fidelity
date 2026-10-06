#!/usr/bin/env bash
# Exp G3 on the v3 stimuli — run this inside an existing GPU shell.
# It does NOT allocate anything: no sbatch, no srun. It assumes a GPU is already visible.
#
#   bash run_v3.sh              # all eight models, resumable
#   bash run_v3.sh molmo2       # one model
#
# Each model appends to results_v3/<model>_expg3.csv and skips items already present, so it is safe to
# interrupt and restart. Compute nodes have no internet, hence the offline flags.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p results_v3 logs

if [ ! -f data/videos_v3/metadata.csv ]; then
  echo "FATAL: data/videos_v3/metadata.csv missing — run generate_v3.py first"; exit 1
fi
if ! .venv-molmo2/bin/python verify_v3.py --meta data/videos_v3/metadata.csv --max-pairs 60; then
  echo "FATAL: verify_v3.py failed — do not spend GPU time on stimuli that do not pass"; exit 1
fi

MODELS=${@:-"molmo2 qwen2.5-vl internvl2.5 videochat-flash llava-next-video video-llama2 qwen2.5-vl-72b internvl2.5-78b"}
echo "V3_START $(date +%Y%m%d_%H%M) host=$(hostname) models=[$MODELS]"
for M in $MODELS; do
  VENV=".venv-${M}"
  [ "$M" = "qwen2.5-vl-72b" ] && VENV=".venv-qwen2.5-vl"
  [ "$M" = "internvl2.5-78b" ] && VENV=".venv-internvl2.5"
  echo "=== ${M} $(date +%H:%M:%S) ==="
  "${VENV}/bin/python" exp_g3.py --model "$M" --out "results_v3/${M}_expg3.csv" \
      > "logs/v3_${M}.log" 2>&1
  echo "STEP_DONE ${M} rc=$? rows=$(($(wc -l < results_v3/${M}_expg3.csv 2>/dev/null || echo 1)-1))"
done
cp data/videos_v3/metadata.csv results_v3/videos_v3_metadata.csv
echo "V3_ALL_DONE $(date +%Y%m%d_%H%M)"
