"""
E5-CONTROL: Shuffled-label control probe for layer-wise analysis.

Trains the same linear probe as layer_probe.py but with randomly permuted
interval labels, providing a null baseline. If the real probe achieves 1.00
but the shuffled probe achieves ~chance, we can confirm the probe detects
interval-specific information rather than any arbitrary variance.

Usage:
  uv run python shuffled_probe.py --models molmo2
  uv run python shuffled_probe.py --models molmo2 video-llama2 --n-shuffles 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from models import load_model
from models.base import VideoModel
from probe import INTERVAL_BINS

ALL_MODELS = [
    "llava-next-video", "video-llama2", "qwen2.5-vl",
    "molmo2", "internvl2.5", "videollama3", "videochat-flash",
]


def log(msg: str = ""):
    print(msg, flush=True)


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
            while arr.dim() > 1:
                arr = arr.mean(0)
            feats[l].append(arr.numpy())
        labels.append(_interval_label(int(row["interval_ms"])))

    return {l: np.stack(v) for l, v in feats.items()}, np.array(labels)


def _probe_shuffled(feats_by_layer: dict, labels: np.ndarray,
                    cv: int, n_shuffles: int, rng: np.random.Generator,
                    out_path: Path, stage: str, model_name: str) -> pd.DataFrame:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    chance = 1 / len(le.classes_)
    log(f"    classes={list(le.classes_)}  chance={chance:.3f}  "
        f"n_samples={len(y)}  n_features={next(iter(feats_by_layer.values())).shape[1]}")

    rows = []
    sorted_layers = sorted(feats_by_layer.items())
    for li, (layer, X) in enumerate(sorted_layers):
        t0 = time.time()
        log(f"    [{li+1}/{len(sorted_layers)}] layer {layer} — fitting real probe...")

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, solver="saga"))
        real_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        log(f"    [{li+1}/{len(sorted_layers)}] layer {layer} — fitting {n_shuffles} shuffled probes...")
        shuf_accs = []
        for si in range(n_shuffles):
            y_shuf = rng.permutation(y)
            clf_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0, solver="saga"))
            s_scores = cross_val_score(clf_s, X, y_shuf, cv=cv, scoring="accuracy", n_jobs=-1)
            shuf_accs.append(s_scores.mean())

        shuf_mean = np.mean(shuf_accs)
        shuf_std = np.std(shuf_accs)
        elapsed = time.time() - t0

        log(f"    layer {layer:3d}: real={real_scores.mean():.3f}  "
            f"shuffled={shuf_mean:.3f}±{shuf_std:.3f}  "
            f"chance={chance:.3f}  ({elapsed:.0f}s)")

        rows.append({
            "layer": layer,
            "real_accuracy": real_scores.mean(),
            "real_std": real_scores.std(),
            "shuffled_mean": shuf_mean,
            "shuffled_std": shuf_std,
            "chance": chance,
            "n_shuffles": n_shuffles,
            "stage": stage,
            "model": model_name,
        })

        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, choices=ALL_MODELS)
    p.add_argument("--meta", default="data/videos/metadata.csv")
    p.add_argument("--stage", default="llm", choices=["llm", "encoder", "both"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-shuffles", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    stages = ["llm", "encoder"] if args.stage == "both" else [args.stage]

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    from models import _CLASSES, _register_lazy
    _register_lazy()

    for model_name in args.models:
        log(f"\n{'='*60}")
        log(f"Model: {model_name}  (shuffled-label control)")
        log(f"{'='*60}")

        model = load_model(model_name)
        df = pd.read_csv(args.meta)
        df = df[df["fps"] == 30]
        if args.limit:
            df = df.groupby("interval_ms", group_keys=False).head(args.limit)

        n = getattr(_CLASSES.get(model_name), "n_llm_layers", 32)
        llm_layers = list(range(0, n, max(1, n // 8)))
        if (n - 1) not in llm_layers:
            llm_layers.append(n - 1)
        enc_layers = list(range(0, 24, 3))

        all_feats = {}
        for stage in stages:
            layers = enc_layers if stage == "encoder" else llm_layers
            log(f"\n  stage={stage}  layers={layers}")

            if stage == "encoder" and type(model).extract_vision_features is VideoModel.extract_vision_features:
                log(f"  ⚠ {model_name}: no vision encoder access — skipping")
                continue

            feats, labels = _collect(model, df, layers, stage)
            all_feats[stage] = (feats, labels)
            log(f"  ✓ features collected for {stage}")

        log(f"\n  Freeing model from GPU...")
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        log(f"  ✓ GPU memory released — starting probes\n")

        for stage, (feats, labels) in all_feats.items():
            out = out_dir / f"{model_name}_shuffled_probe_{stage}.csv"
            log(f"  Probing {stage} ({len(feats)} layers × {args.n_shuffles} shuffles)...")
            _probe_shuffled(feats, labels, args.cv, args.n_shuffles, rng, out, stage, model_name)
            log(f"\n  Saved → {out}")

        del all_feats
        log(f"\n  Done: {model_name}")


if __name__ == "__main__":
    main()
