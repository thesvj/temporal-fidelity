"""
E1-ALT: Prompt sensitivity analysis for temporal order discrimination.

Tests whether the extreme response biases (always YES / always NO) are
prompt-driven by running E1 with alternative prompt formulations.

Three prompt variants:
  - "original":  "Did the {A} move BEFORE the {B}? Answer YES or NO."
  - "reversed":  "Did the {B} move AFTER the {A}? Answer YES or NO."
  - "forced":    "Which shape moved first, the {A} or the {B}? Answer A or B."

Usage:
  uv run python prompt_sensitivity.py --model molmo2
  uv run python prompt_sensitivity.py --model internvl2.5 --model molmo2 --model video-llama2
  uv run python prompt_sensitivity.py --models molmo2 internvl2.5 video-llama2 --limit 20
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models import load_model

PROMPTS = {
    "original": (
        "In this video, did the {color_a} {shape_a} move BEFORE the {color_b} {shape_b}? "
        "Answer YES or NO."
    ),
    "reversed": (
        "In this video, did the {color_b} {shape_b} move AFTER the {color_a} {shape_a}? "
        "Answer YES or NO."
    ),
    "forced": (
        "Which shape moved first in this video, the {color_a} {shape_a} or the "
        "{color_b} {shape_b}? Answer with A or B only."
    ),
}


def _parse_yn(text: str) -> int | None:
    t = text.upper()
    has_yes = bool(re.search(r'\bYES\b', t))
    has_no = bool(re.search(r'\bNO\b', t))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    if has_yes:
        return 1
    return None


def _parse_ab(text: str, color_a: str, shape_a: str, color_b: str, shape_b: str) -> int | None:
    t = text.upper()
    m = re.search(r'\b([AB])\b', t)
    if m:
        return 1 if m.group(1) == "A" else 0
    if color_a.upper() in t or shape_a.upper() in t:
        return 1
    if color_b.upper() in t or shape_b.upper() in t:
        return 0
    return None


def run_variant(model, df: pd.DataFrame, variant: str) -> list[dict]:
    prompt_template = PROMPTS[variant]
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {variant}"):
        prompt = prompt_template.format(**row)
        resp = model.ask(Path(row["path"]), prompt)

        if variant == "forced":
            pred = _parse_ab(resp, row["color_a"], row["shape_a"],
                             row["color_b"], row["shape_b"])
        else:
            pred = _parse_yn(resp)

        gt = int(row["a_first"])
        rows.append({
            **row,
            "variant": variant,
            "response": resp,
            "pred": pred,
            "gt": gt,
            "correct": None if pred is None else int(pred == gt),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   choices=["llava-next-video", "video-llama2", "qwen2.5-vl",
                            "molmo2", "internvl2.5", "videollama3", "videochat-flash"])
    p.add_argument("--meta", default="data/videos/metadata.csv")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--limit", type=int, default=None,
                   help="videos per interval (quick run)")
    p.add_argument("--variants", nargs="+", default=["original", "reversed", "forced"],
                   choices=["original", "reversed", "forced"])
    args = p.parse_args()

    df = pd.read_csv(args.meta)
    df = df[df["fps"] == args.fps]
    if args.limit:
        df = df.groupby("interval_ms", group_keys=False).head(args.limit)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        model = load_model(model_name)

        all_rows = []
        for variant in args.variants:
            rows = run_variant(model, df, variant)
            all_rows.extend(rows)

            vdf = pd.DataFrame(rows).dropna(subset=["pred"])
            n_valid = len(vdf)
            if n_valid > 0:
                p_yes = vdf["pred"].mean()
                acc = vdf["correct"].mean()
                print(f"    {variant}: n={n_valid}, P(YES/A)={p_yes:.3f}, Acc={acc:.3f}")
            else:
                print(f"    {variant}: no parseable responses")

        out = out_dir / f"{model_name}_prompt_sensitivity.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n  Saved → {out}")

        del model


if __name__ == "__main__":
    main()
