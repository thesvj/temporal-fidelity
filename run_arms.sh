#!/usr/bin/env bash
# Frame-alignment arms for Qwen2.5-VL (order question only) — run inside an existing GPU shell; allocates nothing.
#   nohup bash run_arms.sh qwen2.5-vl > logs/arms_7b.out 2>&1 &          # 7B: shapes + footage, four arms
#   MAX_GPU_GIB=126 nohup bash run_arms.sh qwen2.5-vl-72b v3 "shift ts" > logs/arms_72b.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false EXPG3_ONLY=order_real
M=${1:-qwen2.5-vl}; SETS=${2:-"v3 real"}; ARMS=${3:-"harness shift ts native"}
L=logs/run_arms.log; mkdir -p logs
say() { echo "$* $(date +%m%d_%H:%M)" | tee -a $L; }
PY=.venv-qwen2.5-vl/bin/python
for SET in $SETS; do
  META=data/videos_$SET/metadata.csv
  for ARM in $ARMS; do
    A=$ARM; [ "$ARM" = harness ] && A=""
    RES=results_arm_${ARM}_${SET}; mkdir -p $RES
    [ -s $RES/sampled_frames_qwen2.5-vl.jsonl ] || QWEN_ARM=$A $PY record_arm_frames.py --meta $META --out $RES > logs/arm_rec_${ARM}_${SET}.log 2>&1
    LIM=""; [ "$ARM" = harness ] && LIM="--limit 100"      # reproduction check only: the full harness run exists
    for try in 1 2; do
      QWEN_ARM=$A $PY exp_g3.py --model $M --meta $META --blank /nonexistent $LIM --out $RES/${M}_expg3.csv >> logs/arm_${M}_${ARM}_${SET}.log 2>&1 && break
      sleep 20
    done
    say "ARM_DONE $M $SET $ARM rows=$(($(wc -l < $RES/${M}_expg3.csv 2>/dev/null || echo 1)-1)) errors=$(grep -c 'ERROR:' $RES/${M}_expg3.csv 2>/dev/null || true)"
  done
done
say "ARMS_ALL_DONE $M"
