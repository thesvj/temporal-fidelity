#!/usr/bin/env bash
# TempCompass multi-choice at published and harness frame counts — inside an existing GPU shell; allocates nothing.
#   nohup bash run_tc_official.sh > logs/tc_official.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset MAX_GPU_GIB QUANT QWEN_ARM EXPG3_ONLY
L=logs/tc_official.log; mkdir -p logs results_tc_official
for spec in "qwen2.5-vl 64" "qwen2.5-vl 8" "internvl2.5 64" "internvl2.5 8" "llava-next-video 32" "llava-next-video 8" "molmo2 0"; do
  set -- $spec; M=$1; F=$2
  .venv-$M/bin/python tc_official.py --model $M --frames $F --out results_tc_official/${M}_f$F.csv > logs/tc_official_${M}_f$F.log 2>&1
  echo "$(tail -1 logs/tc_official_${M}_f$F.log) rc=$? $(date +%m%d_%H:%M)" | tee -a $L
done
echo "TC_OFFICIAL_ALL_DONE $(date +%m%d_%H:%M)" | tee -a $L
