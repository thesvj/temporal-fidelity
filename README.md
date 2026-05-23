# Temporal Fidelity

Code and data for _"Temporal Fidelity: Probing the Temporal Resolution Limits of Video-Language Models"_ (ACL 2026 Rolling Review).

We test whether Video-LLMs actually perceive time or just exploit response biases. Using synthetic stimuli where timing is the only cue, we apply signal detection theory to decompose accuracy into genuine sensitivity (d') and bias (c), then probe frozen hidden states to find where temporal information lives.

**Main finding:** Most models have near-zero temporal sensitivity despite above-chance accuracy. Temporal information is encoded perfectly in hidden states (probe accuracy = 1.00) but never surfaces in generation (BAcc = 0.50).

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/thesvj/temporal-fidelity.git
cd temporal-fidelity
uv sync
```

Some models need specific `transformers` versions (InternVL2.5 needs 4.45.x, VideoChat-Flash needs 4.40.1, VideoLLaMA3 needs 5.8+). The shell scripts handle this automatically by creating per-model venvs.

## Quick start

Run everything end-to-end (generates stimuli, runs all 7 models, probes layers, produces figures):

```bash
python run_all.py
```

Or run a single model as a smoke test:

```bash
python run_all.py --models molmo2 --quick
```

### Individual steps

**Generate stimuli:**
```bash
python generate.py --out data/videos --n 100 --fps 8 16 30
python generate.py --out data/sim_videos --n 100 --simultaneity
```

**Run behavioral probes (E1–E4):**
```bash
python probe.py --model molmo2 --task order --meta data/videos/metadata.csv
python probe.py --model molmo2 --task interval --meta data/videos/metadata.csv
python probe.py --model molmo2 --task simultaneity --meta data/sim_videos/metadata.csv
python probe.py --model molmo2 --task order --fps 8 --meta data/videos/metadata.csv
```

**Layer probing (E5):**
```bash
python layer_probe.py --model molmo2 --stage both --meta data/videos/metadata.csv
```

**Shuffled-label controls:**
```bash
python shuffled_probe.py --models molmo2 --stage both --n-shuffles 10
```

**Prompt sensitivity:**
```bash
python prompt_sensitivity.py --models molmo2 internvl2.5 qwen2.5-vl video-llama2
```

**Generate figures and tables:**
```bash
python analyze.py                    # everything
python analyze.py --figures-only     # just figures
python analyze.py --tables-only      # just LaTeX tables
python analyze.py --no-bootstrap     # skip CIs (faster)
```

### Full pipeline with per-model venvs

For the full paper results, use the shell scripts which handle transformers version pinning:

```bash
# Generate stimuli first
python generate.py --out data/videos --n 100 --fps 8 16 30
python generate.py --out data/sim_videos --n 100 --simultaneity

# Run all models (creates per-model venvs)
./run.sh

# Shuffled-label controls
./run_shuffled_probe.sh
```

## Models

| Model | Encoder | LLM | HuggingFace ID |
|-------|---------|-----|----------------|
| LLaVA-NeXT-Video | CLIP ViT-L | Mistral-7B | `llava-hf/LLaVA-NeXT-Video-7B-DPO-hf` |
| Video-LLaMA2 | CLIP ViT-L | Mistral-7B | `DAMO-NLP-SG/VideoLLaMA2-7B` |
| Qwen2.5-VL | ViT-mRoPE | Qwen2.5-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Molmo2 | SigLIP2 | Qwen3-8B | `allenai/Molmo2-8B` |
| InternVL2.5 | InternViT-300M | InternLM2-8B | `OpenGVLab/InternVL2_5-8B` |
| VideoLLaMA3 | SigLIP-DiffFP | Qwen2.5-7B | `DAMO-NLP-SG/VideoLLaMA3-7B` |
| VideoChat-Flash | InternVideo2 | Qwen2-7B | `OpenGVLab/VideoChat-Flash-Qwen2-7B_res448` |

## Experiments

- **E1 (Temporal order):** "Did shape A move BEFORE shape B?" across 13 log-spaced intervals (33–10,000 ms)
- **E2 (Interval estimation):** "How much time passed?" with 4 ordinal bins
- **E3 (Simultaneity):** "Did both shapes move at the same time?" at 6 offsets
- **E4 (Frame rate):** E1 repeated at 8, 16, 30 fps
- **E5 (Layer probing):** Logistic regression on frozen hidden states at 8 evenly-spaced layers

## Results

Pre-computed results from the paper are in `results/`. To regenerate figures:

```bash
python analyze.py --figures-only
```

Output goes to `figs/` (PDF) and `results/` (LaTeX tables).

## Project structure

```
├── generate.py              # Stimulus generation
├── probe.py                 # Behavioral probing (E1–E4)
├── layer_probe.py           # Layer-wise diagnostic probing (E5)
├── shuffled_probe.py        # Shuffled-label control probes
├── prompt_sensitivity.py    # Prompt variant analysis
├── analyze.py               # Analysis, figures, and tables
├── run_all.py               # End-to-end pipeline
├── run.sh                   # Full pipeline with per-model venvs
├── run_shuffled_probe.sh    # Shuffled probe pipeline
├── models/
│   ├── base.py              # Base class, video loading, GPU patches
│   ├── molmo2.py
│   ├── qwen25_vl.py
│   ├── internvl25.py
│   ├── llava_next_video.py
│   ├── video_llama2.py
│   ├── videollama3.py
│   └── videochat_flash.py
├── results/                 # Pre-computed CSVs from the paper
└── pyproject.toml
```

## License

MIT
