# Temporal Fidelity: Probing the Temporal Resolution of Video-Language Models

Companion code for the paper *Temporal Fidelity: Probing the Temporal Resolution of Video-Language Models*. Contains the full pipeline — stimulus generation, model inference, layer-wise probing, and analysis — plus the per-trial results behind every number in the paper.

Five controlled psychophysics-inspired experiments measure how well seven open-weight video-language models perceive time in synthetic videos.

## Experiments

| ID | Name | Question / Purpose |
|----|------|--------------------|
| **E001** | Temporal order | "Did shape A move BEFORE shape B?" (YES/NO) |
| **E002** | Interval estimation | "How much time passed?" (A/B/C/D bins) |
| **E003** | Simultaneity detection | "Did both shapes move at the same time?" (YES/NO) |
| **E004** | Frame-rate control | E001 at 8, 16, 30 fps |
| **E005** | Layer-wise probing | Linear probe on frozen features, per layer |
| **E1-ALT** | Prompt sensitivity | Are response biases prompt-driven? |
| **E5-CTRL** | Shuffled-label probe | Null baseline for the layer probe |

All behavioral results are reported with bias-corrected metrics (d', balanced accuracy) alongside raw accuracy.

## Models

| Key | Vision encoder | LLM backbone | HF ID |
|-----|---------------|-------------|-------|
| `llava-next-video` | CLIP ViT-L | Mistral-7B | `llava-hf/LLaVA-NeXT-Video-7B-DPO-hf` |
| `video-llama2` | CLIP ViT-L (Q-Former) | Mistral-7B | `DAMO-NLP-SG/VideoLLaMA2-7B` |
| `qwen2.5-vl` | ViT (mRoPE) | Qwen2.5-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `molmo2` | SigLIP2 | Qwen3-8B | `allenai/Molmo2-8B` |
| `internvl2.5` | InternViT-300M | InternLM2-8B | `OpenGVLab/InternVL2_5-8B` |
| `videollama3` | SigLIP (DiffFP) | Qwen2.5-7B | `DAMO-NLP-SG/VideoLLaMA3-7B` |
| `videochat-flash` | InternVideo2 UMT-L | Qwen2-7B | `OpenGVLab/VideoChat-Flash-Qwen2-7B_res448` |

## Quickstart

Requires one NVIDIA GPU (>= 48 GB VRAM), CUDA >= 12.8, Python >= 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thesvj/temporal-fidelity.git && cd temporal-fidelity

# 1. Generate stimuli (one-time, ~5 min)
uv run python generate.py --out data/videos --n 100 --fps 8 16 30
uv run python generate.py --out data/sim_videos --n 100 --fps 30 --simultaneity

# 2. Run all 7 models end-to-end
./run.sh

# 3. Tables land in results/, figures in figs/
```

Full instructions — per-model runs, transformers version pins, robustness experiments, output formats — are in **[REPRODUCTION.md](REPRODUCTION.md)**.

## Layout

```
generate.py              Synthetic video stimulus generation
probe.py                 E001-E004: model inference -> per-video results CSV
layer_probe.py           E005: logistic regression on frozen layer features
prompt_sensitivity.py    E1-ALT: prompt variant robustness
shuffled_probe.py        E5-CTRL: shuffled-label null baseline
analyze.py               Figures, tables, SDT analysis, bootstrap CIs
run_all.py / run.sh      End-to-end orchestration (per-model isolated venvs)
run_shuffled_probe.sh    E5-CTRL across models
models/                  One adapter per model (VideoModel ABC in base.py)
results/                 Per-trial CSVs + LaTeX tables (committed)
data/, figs/, logs/      Generated stimuli, figures, run logs (not committed)
```
