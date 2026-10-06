#!/usr/bin/env bash
#SBATCH -J tc_margin
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o logs/tc_margin_%j.out
#SBATCH -e logs/tc_margin_%j.err
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
echo "TCM_START $(date +%Y%m%d_%H%M) host=$(hostname)"
for M in molmo2 qwen2.5-vl internvl2.5 videochat-flash llava-next-video video-llama2 qwen2.5-vl-72b; do
  VENV=".venv-${M}"; [ "$M" = "qwen2.5-vl-72b" ] && VENV=".venv-qwen2.5-vl"
  echo "=== ${M} $(date +%H:%M:%S) ==="
  "${VENV}/bin/python" tc_margin.py --model "$M" --out "results/${M}_tc_margin.csv" > "logs/tcm_${M}.log" 2>&1 &
  wait $!
  echo "STEP_DONE ${M} rc=$? rows=$(($(wc -l < results/${M}_tc_margin.csv 2>/dev/null || echo 1)-1))"
done
echo "TCM_ALL_DONE $(date +%Y%m%d_%H%M)"
