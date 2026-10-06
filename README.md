# Chance Is Not Blindness — code, stimuli generator and per-trial outputs

Anonymous release for the ARR submission. Everything the paper reports is computed from the CSVs in
`results_*` by the analysis scripts below, on CPU. Re-running the models needs GPUs.

## Stimuli (v3)

    python generate_v3.py --out data/videos_v3 --n 100 --seed 0 --twins data/twin_v3 --pairs 240
    python verify_v3.py --meta data/videos_v3/metadata.csv --max-pairs 60
    python gen_blank.py

1,200 lossless clips (480x480, 30 fps): two shapes each jump once; 12 inter-jump intervals (67 ms–10 s);
colour, form, size, position, direction, lead-in, tail and background randomized per trial; the question
names either shape first (`name_first`). 240 matched sets add a frozen twin (no motion) and one-shape-moves
clips. `verify_v3.py` checks uniqueness, label balance, independence of the label from every nuisance, and
bit-identity of decoded frames outside the jump window for order-flipped pairs. Per-trial metadata:
`results_v3/videos_v3_metadata.csv`.

## Models

`models/` holds one adapter per model (Molmo2, VideoChat-Flash, Qwen2.5-VL 7B/72B, InternVL2.5 8B/78B,
LLaVA-NeXT-Video, VideoLLaMA2). Each model runs in its own virtual environment (`.venv-<model>`); 8 uniformly
sampled frames unless stated. The frame indices each adapter actually decodes are logged by
`record_sampled_frames.py` (`results_v3/sampled_frames_*.jsonl`); visibility of the order is computed
from these, not from the requested frame count.

## Controlled experiments

    bash run_v3.sh [model ...]         # exp_g3.py: order, paraphrase, forced choice, 32 frames, interval,
                                       # move/still on real and frozen twins, one-shape-moves, blank
    python steer_v3.py --model qwen2.5-vl --mode ablate --controls --layers 21 --n-test 400 \
        --out results_v3/ablate2_qwen2.5-vl.csv
    python longpaper/extract_order_feats.py --model qwen2.5-vl && python longpaper/probe_order.py --model qwen2.5-vl

Every answer is stored with the first-token YES−NO logit margin. Outputs: `results_v3/<model>_expg3.csv`
(column `item` = condition), `results_v3/steer_*.csv`, `results_v3/ablate*_*.csv`, `longpaper/results_v3/`.

## TempCompass audit

    python make_tc_controls.py                     # static (one frame) and blank control clips
    sbatch run_tc_yesno.sbatch; sbatch run_tc_margin.sbatch

TempCompass videos are not redistributed; download them from the TempCompass release into
`data/tempcompass/`. Outputs: `results_long/*_tc_yesno.csv`, `results_long/*_tc_margin.csv` (real clips),
`results_tc/*_tc_margin_{static,blank}.csv`.

## Analysis (CPU)

    uv run --no-project --with numpy python analyze_v3.py --boot 10000     # Table 1, stage rule, Holm
    uv run --no-project --with numpy python robust_v3.py                   # same-answer AUC, recency, phrasing
    uv run --no-project --with numpy --with pandas python appendix_v3.py
    uv run --no-project --with numpy --with pandas python tc_controls.py
    python steer_summary.py

`results_v3/analysis_8models_b10k.txt` is the output behind the submitted Table 1 (the paper's table was assembled from it by hand: P(YES) and d' on visible trials; `analyze_v3.py` also writes `table_main_rows.tex`, which prints c instead and is not the submitted table); `profile_v3.py` writes the loss budget, matched-criterion, per-phrasing and frozen-twin numbers (`results_v3/profile_v3.json`).
`results_long/*_expg.csv` and `*_expg2.csv` are from the earlier v1/v2 stimulus sets (probe results on those sets are not shipped), kept for the
withdrawn results discussed in the appendix; they are not used for any main-text number.

## Composited real footage (Appendix "Composited Real Footage with Controlled Onsets")

The same trial as the shape stimuli, built from real footage: two stacked 480x240 panels, each still, then
15 played frames of a Wikimedia Commons video, then still. Every trial is rendered in both orders.

    python fetch_clips.py search --out data/real_src && python fetch_clips.py download --out data/real_src
    uv run --no-project --with numpy --with opencv-python-headless python screen_clips.py --src data/real_src
    # clips.csv is the hand-picked result of that screening (41 bursts; authors and licences in SOURCES_real.md)
    python generate_real.py --clips clips.csv --src data/real_clips --out data/videos_real --n 50 --seed 0 \
        --twins data/twin_real --n-twins 240
    bash run_real.sh          # inside a GPU shell: record decoded frames, verify, six 7-8B models
    uv run --no-project --with numpy python analyze_real.py --res results_real --boot 2000

`verify_real.py` must pass before any model runs (bit-identity of order-swapped pairs outside the event
window, frozen twins, an ideal observer on the logged frames). Outputs: `results_real/<model>_expg3.csv`,
`results_real/numbers_real.json`, `results_real/evidence.csv` (trial class and evidence strength per sampler).
The composited clips are derived works released under CC BY-SA 4.0. `results_pilot/` is a 120-trial pilot on
an earlier 29-burst clip list (`clips_pilot.csv`) and is not used for any reported number.

## Adapter validation (Appendix L)
- `tc_official.py`: TempCompass multi-choice with the official prompt suffix and matching rules (`results_tc_official/`).
- `mvbench_prep.py`, `mvbench_eval.py`: MVBench (19 of 20 tasks; NTU pose videos are not distributed) for
  VideoChat-Flash (official call, and our adapter's own call on the 14 unbounded tasks) and Video-LLaMA2
  (`results_mvbench/`). Data: Hugging Face `OpenGVLab/MVBench` (gated) under `data/mvbench`.

## LLaVA-NeXT-Video system prompt (2026-10-06)
The Hugging Face chat template for `llava-hf/LLaVA-NeXT-Video-7B-DPO-hf` omits LLaVA's vicuna_v1 system prompt. All LLaVA-NeXT-Video
results in the paper were run with it (`LLAVA_SYS=1`, see `models/llava_next_video.py`). Runs with the template's default prompt are
kept as `*.hftemplate.*`. Validation: `tc_official.py` + `tc_judge.py` (unmatched multiple-choice answers scored by a local judge, as on
the TempCompass leaderboard); results in `results_tc_official/`. Frame-delivery check on TempCompass: `tc_arms.py`, `results_tc_arms/`.
Within-class probe vs margin: `longpaper/within_class.py`, `longpaper/ablation_by_class.py`; within-video TempCompass AUC: `longpaper/tc_within.py`.
