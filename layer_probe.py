"""
E005 — Layer-wise diagnostic probing.

Trains a logistic regression at each layer to predict temporal interval class
from frozen hidden-state features.  Runs against both the LLM backbone and
the vision encoder to test H4 (bottleneck is in the encoder, not the LLM).

Usage:
  # LLM layers (default)
  uv run python layer_probe.py --model llava-next-video

  # Vision encoder layers
  uv run python layer_probe.py --model llava-next-video --stage encoder --layers 0 4 8 12 16 20 23

  # Both stages, save to named prefix
  uv run python layer_probe.py --model llava-next-video --stage both
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from models import load_model
from models.base import VideoModel
from probe import INTERVAL_BINS

ALL_MODELS = ["llava-next-video", "video-llama2", "qwen2.5-vl", "molmo2", "internvl2.5", "videollama3", "videochat-flash"]


def _interval_label(ms: int) -> str:
    for lo, hi, letter in INTERVAL_BINS:
        if lo <= ms < hi:
            return letter
    return "D"


def _collect(model, meta: pd.DataFrame, layers: list[int], stage: str) -> tuple[dict, np.ndarray]:
    feats = {l: [] for l in layers}
    labels = []

    has_vision = type(model).extract_vision_features is not VideoModel.extract_vision_features
    extract = (
        model.extract_vision_features
        if stage == "encoder" and has_vision
        else model.extract_features
    )

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"  {stage} features"):
        hidden = extract(Path(row["path"]), layers)
        for l, h in hidden.items():
            arr = h.float()
            while arr.dim() > 1:   # mean-pool batch/seq/time dims → (hidden_dim,)
                arr = arr.mean(0)
            feats[l].append(arr.numpy())
        labels.append(_interval_label(int(row["interval_ms"])))

    return {l: np.stack(v) for l, v in feats.items()}, np.array(labels)


def _probe(feats_by_layer: dict, labels: np.ndarray, cv: int) -> pd.DataFrame:
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    chance = 1 / len(le.classes_)
    print(f"    classes={list(le.classes_)}  chance={chance:.3f}")

    rows = []
    for layer, X in sorted(feats_by_layer.items()):
        clf    = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        print(f"    layer {layer:3d}: {scores.mean():.3f} ± {scores.std():.3f}")
        rows.append({"layer": layer, "accuracy": scores.mean(), "std": scores.std()})

    return pd.DataFrame(rows)


def run(model_name: str, meta_path: Path, stages: list[str],
        llm_layers: list[int], enc_layers: list[int],
        cv: int, out_prefix: str, limit: int | None):

    model = load_model(model_name)
    df    = pd.read_csv(meta_path)
    df    = df[df["fps"] == 30]   # standardize on 30 fps for E005 to remove fps as a confound
    if limit:
        df = df.groupby("interval_ms", group_keys=False).head(limit)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    for stage in stages:
        layers = enc_layers if stage == "encoder" else llm_layers
        print(f"\n[{model_name}] stage={stage}  layers={layers}")

        if stage == "encoder" and type(model).extract_vision_features is VideoModel.extract_vision_features:
            print(f"  ⚠ {model_name} does not expose vision encoder — skipping encoder stage")
            continue

        feats, labels = _collect(model, df, layers, stage)
        results = _probe(feats, labels, cv)
        results["stage"] = stage

        out = out_dir / f"{out_prefix}_{stage}_layers.csv"
        results.to_csv(out, index=False)
        print(f"  Saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True, choices=ALL_MODELS)
    p.add_argument("--meta",       default="data/videos/metadata.csv")
    p.add_argument("--stage",      default="llm", choices=["llm", "encoder", "both"])
    p.add_argument("--llm-layers", nargs="+", type=int, default=None,
                   help="LLM layer indices to probe (default: 8 evenly-spaced layers up to model max)")
    p.add_argument("--enc-layers", nargs="+", type=int, default=list(range(0, 24, 3)))
    p.add_argument("--cv",         type=int, default=5)
    p.add_argument("--out",        default=None, help="output filename prefix (default: model name)")
    p.add_argument("--limit",      type=int, default=None)
    args = p.parse_args()

    stages     = ["llm", "encoder"] if args.stage == "both" else [args.stage]
    out_prefix = args.out or args.model

    # Use model-aware default if --llm-layers not specified
    if args.llm_layers is None:
        from models import _CLASSES, _register_lazy
        _register_lazy()
        n = getattr(_CLASSES.get(args.model), "n_llm_layers", 32)
        llm_layers = list(range(0, n, max(1, n // 8)))
        if (n - 1) not in llm_layers:
            llm_layers.append(n - 1)   # always include the last layer
    else:
        llm_layers = args.llm_layers

    run(args.model, Path(args.meta), stages,
        llm_layers, args.enc_layers,
        args.cv, out_prefix, args.limit)
