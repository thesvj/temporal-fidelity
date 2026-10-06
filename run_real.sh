#!/usr/bin/env bash
# Real-footage run, end to end, unattended — run this inside an existing GPU shell.
# It does NOT allocate anything: no sbatch, no srun. It assumes one GPU is already visible.
#
#   nohup bash run_real.sh > logs/run_real.out 2>&1 &
#
# Steps: render (skipped if metadata exists) -> record the frames each loader decodes -> verify on those
# frames (refuses to go on if a check fails) -> six 7-8B models, PAR at a time on the one GPU. Every step
# appends one line to logs/run_real.log; REAL_ALL_DONE is the last and says how many models are complete.
# Each model appends to results_real/<model>_expg3.csv and skips items already present, so it is safe to
# interrupt and restart.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset MAX_GPU_GIB QUANT          # adapter switches used for the 72-78B runs; must be off here
PAR=${PAR:-2}
export OMP_NUM_THREADS=$((20 / PAR))
OUT=data/videos_real; TW=data/twin_real; RES=results_real
L=logs/run_real.log
mkdir -p $RES logs
say() { echo "$* $(date +%m%d_%H:%M)" | tee -a $L; }
trap 'trap - TERM; say REAL_ABORTED; kill -- -$$' INT TERM
PY=.venv-molmo2/bin/python

say "REAL_START host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"
if [ ! -f $OUT/metadata.csv ]; then
  $PY generate_real.py --clips clips.csv --src data/real_clips --out $OUT --n 50 --seed 0 \
      --twins $TW --n-twins 240 > logs/real_render.log 2>&1 || { say "FATAL render"; exit 1; }
fi
NTRIAL=$(($(wc -l < $OUT/metadata.csv) - 1))
say "RENDER_DONE trials=$NTRIAL size=$(du -sh $OUT | cut -f1)"

# frames each loader really decodes (CPU). All four at once; a missing or short log is fatal, because the
# analysis would otherwise fall back to a re-implementation of the sampler without saying so.
for M in molmo2 qwen2.5-vl video-llama2 videochat-flash; do
  F=$RES/sampled_frames_$M.jsonl
  if [ ! -s $F ] || [ "$(wc -l < $F)" -ne "$NTRIAL" ]; then
    rm -f $F
    .venv-$M/bin/python record_sampled_frames.py --model $M --meta $OUT/metadata.csv --out $F > logs/real_rec_$M.log 2>&1 &
  fi
done
wait
for M in molmo2 qwen2.5-vl video-llama2 videochat-flash; do
  [ "$(wc -l < $RES/sampled_frames_$M.jsonl 2>/dev/null || echo 0)" -eq "$NTRIAL" ] || { say "FATAL record $M — see logs/real_rec_$M.log"; exit 1; }
done
cp $OUT/metadata.csv $RES/videos_v3_metadata.csv
say "RECORD_DONE 4 samplers x $NTRIAL clips"

if ! $PY verify_real.py --meta $OUT/metadata.csv --res $RES > logs/real_verify.log 2>&1; then
  say "FATAL verify failed — see logs/real_verify.log; no GPU time spent"; exit 1
fi
cp $OUT/evidence.csv $RES/evidence.csv
say "VERIFY_DONE $(tail -1 logs/real_verify.log)"

NTWIN=$(ls $TW | wc -l)
NBLANK=$(grep -c ',blank,' data/blank_videos/metadata.csv || true)
EXPECT=$((NTRIAL + 5 * NTWIN + NBLANK))
run_one() {
  local M=$1 rc rows err try C=$RES/${M}_expg3.csv
  for try in 1 2 3; do
    timeout -k 60 5h .venv-$M/bin/python exp_g3.py --model $M --meta $OUT/metadata.csv \
        --blank data/blank_videos/metadata.csv --out $C >> logs/real_$M.log 2>&1; rc=$?
    rows=$(( $([ -f $C ] && wc -l < $C || echo 1) - 1 )); err=$(grep -c 'ERROR:' $C 2>/dev/null || true)
    [ $rc -eq 0 ] && [ $rows -eq $EXPECT ] && [ "${err:-0}" -eq 0 ] && break
    sleep 30
  done
  say "STEP_DONE $M rc=$rc rows=$rows/$EXPECT errors=${err:-NA} tries=$try"
}

MODELS=${@:-"molmo2 qwen2.5-vl videochat-flash internvl2.5 llava-next-video video-llama2"}
for M in $MODELS; do
  run_one $M &
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
done
wait
say "REAL_ALL_DONE complete=$(grep -c "rows=$EXPECT/$EXPECT errors=0" $L)/6"
