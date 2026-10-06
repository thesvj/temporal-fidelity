#!/usr/bin/env bash
# Table 1 under each model's reference preprocessing (plan: ../reviews/mainpush_1003/plan_ref.md).
# Runs inside the existing GPU shell (tmux vg, job 5016); allocates nothing.
#   nohup bash run_ref.sh > logs/run_ref.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset MAX_GPU_GIB QUANT QWEN_ARM INTERNVL_REF EXPG3_ONLY
L=logs/run_ref.log; mkdir -p logs
say() { echo "$* $(date +%m%d_%H:%M)" | tee -a $L; }
T1=order_real,order_rev,order_forced,moved_real,moved_twin,moved_solo_hit,moved_solo_fa,still_real,still_twin,order_twin,order_blank
one() {  # model env-assignments set results-dir items [timeout]
  local M=$1 ENVV=$2 SET=$3 RES=$4 ONLY=$5 TO=${6:-14h} C=$4/$1_expg3.csv rc
  for try in 1 2 3; do
    env $ENVV EXPG3_ONLY=$ONLY timeout -k 60 $TO .venv-${M%-72b}/bin/python exp_g3.py --model $M \
        --meta data/videos_$SET/metadata.csv --blank data/blank_videos/metadata.csv --out $C >> logs/ref_${M}_$SET.log 2>&1; rc=$?
    [ $rc -eq 0 ] && ! grep -q 'ERROR:' $C && break; sleep 30
  done
  say "REF_DONE $M $SET rc=$rc rows=$(($(wc -l < $C 2>/dev/null || echo 1)-1)) errors=$(grep -c 'ERROR:' $C 2>/dev/null || true)"
}
for SET in v3 real; do
  .venv-qwen2.5-vl/bin/python record_ref_frames.py --kind qwen  --meta data/videos_$SET/metadata.csv --out results_ref_qwen_$SET >> $L 2>&1
  .venv-internvl2.5/bin/python record_ref_frames.py --kind ivl32 --meta data/videos_$SET/metadata.csv --out results_ref_ivl_$SET >> $L 2>&1
  N=$(($(wc -l < data/videos_$SET/metadata.csv)-1))
  for K in qwen ivl; do
    [ "$(wc -l < results_ref_${K}_$SET/sampled_frames_qwen2.5-vl.jsonl 2>/dev/null || echo 0)" -eq $N ] || { say "FATAL frame log $K $SET"; exit 1; }
  done
done
say "REF_FRAMES_OK"
REALONLY=order_real,order_rev,moved_real,moved_twin,order_twin,bind_twin,order_blank
CORE=order_real,moved_real,moved_twin          # what the stage rule needs; first, in case time runs out
# A: core items, 7B and InternVL together
one qwen2.5-vl QWEN_ARM=native v3 results_ref_qwen_v3 $CORE &
one internvl2.5 INTERNVL_REF=1 v3 results_ref_ivl_v3 $CORE &
wait; say "REF_PHASE_A_DONE"
# B: 72B core, GPU alone
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; do sleep 60; done
one qwen2.5-vl-72b "QWEN_ARM=native MAX_GPU_GIB=126" v3 results_ref_qwen_v3 $CORE 8h
say "REF_PHASE_B_DONE"
# C: the rest of Table 1 (exp_g3 resumes: rows already written are skipped)
one qwen2.5-vl QWEN_ARM=native v3 results_ref_qwen_v3 $T1 &
one internvl2.5 INTERNVL_REF=1 v3 results_ref_ivl_v3 $T1 &
wait; say "REF_PHASE_C_DONE"
# D: footage
one qwen2.5-vl QWEN_ARM=native real results_ref_qwen_real $REALONLY &
one internvl2.5 INTERNVL_REF=1 real results_ref_ivl_real $REALONLY &
wait
say "REF_ALL_DONE"
