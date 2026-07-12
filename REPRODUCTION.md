# Reproduction Guide

Step-by-step instructions to reproduce every experiment in the paper from scratch. See the [README](README.md) for an overview of the experiments, models, and outputs.

## Requirements

- **GPU**: 1x NVIDIA GPU with >= 48 GB VRAM (tested on B200 183 GB)
- **CUDA**: >= 12.8
- **Python**: >= 3.12
- **[uv](https://docs.astral.sh/uv/)**: package manager (handles venvs automatically)
- **Disk**: ~15 GB for model weights (cached by HuggingFace), ~2 GB for generated videos

## 1. Generate synthetic video stimuli

```bash
# Main dataset: 13 intervals x 3 fps x 100 videos = 3900 videos
uv run python generate.py --out data/videos --n 100 --fps 8 16 30

# Simultaneity dataset: 6 offsets x 100 videos = 600 videos
uv run python generate.py --out data/sim_videos --n 100 --fps 30 --simultaneity
```

**Stimulus design.** Each video shows two colored shapes (e.g., red square, blue circle) on a gray background. After a 1-second lead-in, the shapes move sequentially with a controlled inter-event interval. Intervals are frame-aligned at 30 fps and log-spaced to span the range from near-instantaneous (33 ms = 1 frame) to clearly separated (10 s):

```
33  67  100  200  333  500  1000  1500  2000  3000  5000  7000  10000  ms
```

Simultaneity offsets (E003): `0  33  67  100  200  333` ms.

Frame alignment ensures every interval produces a visually distinct stimulus at 30 fps. At lower fps (E004), small intervals collapse to same-frame, revealing how frame rate limits temporal resolution.

## 2. Run experiments

### All models (recommended)

```bash
./run.sh                                    # all 7 models, full dataset
./run.sh llava-next-video qwen2.5-vl        # specific models only
LIMIT=10 ./run.sh                           # cap at 10 videos per condition (fast test)
SKIP_LAYERS=1 ./run.sh                      # skip E005 layer probes (saves ~1hr/model)
```

Each model runs in an isolated venv (`.venv-<model>`) managed by `UV_PROJECT_ENVIRONMENT`. Models requiring specific `transformers` versions are pinned automatically:

| Model | transformers version |
|-------|---------------------|
| `internvl2.5` | `>=4.45, <4.46` |
| `videollama3` | `>=5.8` |
| `videochat-flash` | `==4.40.1` |
| all others | `>=4.49, <5` (from lockfile) |

### Single model (development)

```bash
# Set the venv for a specific model
export UV_PROJECT_ENVIRONMENT=.venv-llava-next-video
uv sync

# E001: temporal order
uv run python probe.py --model llava-next-video --task order --meta data/videos/metadata.csv --fps 30

# E002: interval estimation
uv run python probe.py --model llava-next-video --task interval --meta data/videos/metadata.csv --fps 30

# E003: simultaneity
uv run python probe.py --model llava-next-video --task simultaneity --meta data/sim_videos/metadata.csv

# E004: frame-rate control (runs order at 8 and 16 fps; 30 fps reuses E001)
uv run python probe.py --model llava-next-video --task order --meta data/videos/metadata.csv --fps 8
uv run python probe.py --model llava-next-video --task order --meta data/videos/metadata.csv --fps 16

# E005: layer-wise probing (LLM + vision encoder)
uv run python layer_probe.py --model llava-next-video --meta data/videos/metadata.csv --stage both

# Analysis
uv run python analyze.py
```

### Smoke test (~2 min)

```bash
uv run python run_all.py --quick --models llava-next-video
```

## 3. Robustness experiments

These test whether the headline results are robust to prompt wording and to probe artifacts.

### E1-ALT: Prompt sensitivity

Tests whether the extreme response biases (always-YES / always-NO) are driven by prompt wording rather than genuine model behavior. Runs E1 with three prompt variants:

- **original**: "Did the {A} move BEFORE the {B}? Answer YES or NO."
- **reversed**: "Did the {B} move AFTER the {A}? Answer YES or NO."
- **forced**: "Which shape moved first, the {A} or the {B}? Answer A or B."

```bash
# Run on key models: one success (molmo2), one always-NO (internvl2.5), one always-YES (video-llama2)
UV_PROJECT_ENVIRONMENT=.venv-molmo2 uv run python prompt_sensitivity.py \
  --models molmo2 --limit 20

UV_PROJECT_ENVIRONMENT=.venv-internvl2.5 uv run python prompt_sensitivity.py \
  --models internvl2.5 --limit 20

UV_PROJECT_ENVIRONMENT=.venv-video-llama2 uv run python prompt_sensitivity.py \
  --models video-llama2 --limit 20
```

If biased models remain biased across all prompt variants, the bias is model-intrinsic (strong evidence). If forced-choice breaks the bias, the YES/NO format is part of the problem.

### E5-CTRL: Shuffled-label control probe

Trains the same linear probe as E005 but with randomly permuted interval labels. If the real probe achieves 1.00 but the shuffled probe achieves ~0.25 (chance for 4 classes), the probe is detecting interval-specific temporal information, not arbitrary feature variance.

```bash
# Run on 2 models: one high-probe (molmo2), one low-probe (video-llama2)
UV_PROJECT_ENVIRONMENT=.venv-molmo2 uv run python shuffled_probe.py \
  --models molmo2 --stage both --n-shuffles 10

UV_PROJECT_ENVIRONMENT=.venv-video-llama2 uv run python shuffled_probe.py \
  --models video-llama2 --stage llm --n-shuffles 10
```

## 4. Analysis

```bash
uv run python analyze.py                   # all experiments
uv run python analyze.py --task order      # single experiment
```

LaTeX tables are written to `results/` and figures to `figs/`.

`run_all.py` runs the core pipeline for one or more models:

```
generate.py  -->  probe.py (E001-E004)  -->  layer_probe.py (E005)  -->  analyze.py
                                        \
                                         +-> prompt_sensitivity.py (E1-ALT)
                                         +-> shuffled_probe.py (E5-CTRL)
```

## Output files (`results/`)

| File | Description |
|------|-------------|
| `{model}_{task}.csv` | Per-video raw results (response, prediction, ground truth, correct) |
| `{model}_order_fps{N}.csv` | E004 frame-rate results |
| `{model}_{llm\|encoder}_layers.csv` | E005 layer probe accuracy per layer |
| `{model}_prompt_sensitivity.csv` | E1-ALT prompt variant results (original, reversed, forced) |
| `{model}_shuffled_probe_{stage}.csv` | E5-CTRL real vs shuffled-label probe accuracy per layer |
| `summary_{task}.csv` | Model x interval accuracy table |
| `thresholds_{task}.csv` | Psychometric fit parameters (threshold, slope) |
| `response_bias.csv` | Per-model response bias (P(YES) for order/simultaneity, P(A/B/C/D) for interval) |
| `table{1-5}_*.tex` | Publication-ready LaTeX tables (booktabs) |

## Design decisions

**Frame-aligned intervals.** At 30 fps each frame spans 33.3 ms. Intervals that don't align to frame boundaries produce identical stimuli — e.g., 10 ms and 30 ms both yield 0-frame separation. All intervals are multiples of ~33 ms so every condition is visually distinct at the reference rate.

**Log-spaced scale.** Temporal perception follows Weber's law (logarithmic sensitivity). Log-spacing concentrates stimuli where discrimination is most interesting: near the resolution boundary.

**Wilson confidence intervals.** Standard normal-approximation CIs can extend below 0 or above 1 for proportions near boundaries. Wilson score intervals have correct coverage at all sample sizes.

**Response bias check.** If a model always says "YES" or always picks "(A)", high accuracy on some conditions is an artifact, not temporal perception. `response_bias.csv` reports marginal response distributions per model to detect this.

**Per-model venvs.** The 7 models span transformers 4.40 to 5.8+. A single environment cannot satisfy all constraints. `UV_PROJECT_ENVIRONMENT` isolates each model cleanly.

**E005 standardized to 30 fps.** Layer probes use only 30 fps videos to remove frame rate as a confound — the probe should measure temporal information in the representation, not frame-counting artifacts.

## Adding a new model

1. Create `models/<name>.py` with a class inheriting from `VideoModel`
2. Implement `ask()` (required), optionally `extract_features()` and `extract_vision_features()`
3. Set class attributes `n_llm_layers` and `n_enc_layers`
4. Add the HF ID to `HF_IDS` in `models/base.py`
5. Register the class in `models/__init__.py`
6. Add the model key to `ALL_MODELS` in `run_all.py` and `run.sh`
7. If the model needs a specific transformers version, add it to `TF_PIN` in `run.sh`
