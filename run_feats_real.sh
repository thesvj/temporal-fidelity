#!/usr/bin/env bash
# Last-token states of the order-question pass on the real-footage trials — inside an existing GPU shell.
#   nohup bash run_feats_real.sh > logs/feats_real.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset MAX_GPU_GIB QUANT QWEN_ARM EXPG3_ONLY
PAR=${PAR:-2}; L=logs/feats_real.log; mkdir -p logs longpaper/feats_real
say() { echo "$* $(date +%m%d_%H:%M)" | tee -a $L; }
one() {
  local M=$1
  [ -f longpaper/feats_real/$M.npz ] || .venv-$M/bin/python longpaper/extract_order_feats.py --model $M \
      --meta data/videos_real/metadata.csv --conds real --outdir longpaper/feats_real > logs/feats_real_$M.log 2>&1
  say "FEATS_DONE $M rc=$? $(ls -la longpaper/feats_real/$M.npz 2>/dev/null | awk '{print $5}')"
}
for M in ${@:-"qwen2.5-vl molmo2 internvl2.5 videochat-flash"}; do
  one $M &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
done
wait
say "FEATS_ALL_DONE"
