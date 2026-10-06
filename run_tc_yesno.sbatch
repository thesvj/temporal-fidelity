#!/usr/bin/env bash
#SBATCH -J tc_yesno
#SBATCH -p <partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@180
#SBATCH -o logs/tc_yesno_%j.out
#SBATCH -e logs/tc_yesno_%j.err
# Exp F: TempCompass yes/no, real/static/blank. Resumable per item (tc_yesno.py appends + skips done).
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # compute nodes have no internet; use local cache
cd "$SLURM_SUBMIT_DIR"
trap 'echo "USR1 walltime signal $(date +%H:%M:%S) — rows already flushed; resubmit to resume"' USR1
echo "TCYN_START $(date +%Y%m%d_%H%M) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

.venv-molmo2/bin/python make_tc_controls.py || { echo "CONTROLS_FAIL"; exit 1; }

# small models first (cheapest, most needed), 72B last so a timeout only loses the scale point
for M in molmo2 qwen2.5-vl internvl2.5 videochat-flash llava-next-video video-llama2 qwen2.5-vl-72b; do
  VENV=".venv-${M}"; [ "$M" = "qwen2.5-vl-72b" ] && VENV=".venv-qwen2.5-vl"
  echo "=== ${M} $(date +%H:%M:%S) ==="
  "${VENV}/bin/python" tc_yesno.py --model "$M" --out "results/${M}_tc_yesno.csv" \
    > "logs/tc_yesno_${M}.log" 2>&1 &
  wait $!
  echo "STEP_DONE ${M} rc=$? rows=$(($(wc -l < results/${M}_tc_yesno.csv 2>/dev/null || echo 1)-1))"
done
echo "TCYN_ALL_DONE $(date +%Y%m%d_%H%M)"
