"""
Probing tasks for E001, E002, E003, E004.

  E001 — temporal order:   --task order
  E002 — interval est.:    --task interval
  E003 — simultaneity:     --task simultaneity
  E004 — frame-rate ctrl:  --task order --fps 8   (or 16, 30)

Results written to results/<model>_<task>[_fps<N>].csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models import load_model

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = {
    "order": (
        "In this video, did the {color_a} {shape_a} move BEFORE the {color_b} {shape_b}? "
        "Answer YES or NO."
    ),
    "interval": (
        "How much time passed between the two events? "
        "Answer exactly one letter: "
        "(A) Less than 1 second, "
        "(B) 1–2 seconds, "
        "(C) 2–5 seconds, "
        "(D) More than 5 seconds."
    ),
    "simultaneity": (
        "Did both shapes start moving at EXACTLY the same time? Answer YES or NO."
    ),
}

INTERVAL_BINS = [(0, 1000, "A"), (1000, 2000, "B"), (2000, 5000, "C"), (5000, 1e9, "D")]


def _interval_label(ms: int) -> str:
    for lo, hi, letter in INTERVAL_BINS:
        if lo <= ms < hi:
            return letter
    return "D"


def _parse_yn(text: str) -> int | None:
    t = text.upper()
    has_yes = bool(re.search(r'\bYES\b', t))
    has_no  = bool(re.search(r'\bNO\b',  t))
    if has_yes and not has_no: return 1
    if has_no  and not has_yes: return 0
    if has_yes: return 1   # both present: YES wins (model affirmed then qualified)
    return None


def _parse_abcd(text: str) -> str | None:
    t = text.upper()
    # Explicit parenthesized: (A), (B), (C), (D)
    m = re.search(r'\(([ABCD])\)', t)
    if m:
        return m.group(1)
    # Answer-indicator keyword: "Answer: B", "the answer is B"
    m = re.search(r'\bANSWER\s*(?:IS|:)\s*([ABCD])\b', t)
    if m:
        return m.group(1)
    # First standalone letter by document position
    m = re.search(r'\b([ABCD])\b', t)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Core probing loop
# ---------------------------------------------------------------------------

def run(model_name: str, meta: Path, task: str, fps_filter: int | None, out: Path, limit: int | None):
    model = load_model(model_name)
    df = pd.read_csv(meta)

    if fps_filter is not None:
        df = df[df["fps"] == fps_filter]
    if limit:
        df = df.groupby("interval_ms", group_keys=False).head(limit)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_name}/{task}"):
        prompt = PROMPTS[task].format(**row) if task == "order" else PROMPTS[task]

        resp = model.ask(Path(row["path"]), prompt)

        if task in ("order", "simultaneity"):
            pred = _parse_yn(resp)
            gt   = int(row["a_first"]) if task == "order" else int(row["simultaneous"])
        else:
            pred = _parse_abcd(resp)
            gt   = _interval_label(int(row["interval_ms"]))

        rows.append({
            **row,
            "response": resp,
            "pred":     pred,
            "gt":       gt,
            "correct":  None if pred is None else int(pred == gt),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nResults → {out}  (n={len(rows)})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True,
                   choices=["llava-next-video", "video-llama2",
                            "qwen2.5-vl", "molmo2", "internvl2.5",
                            "videollama3", "videochat-flash"])
    p.add_argument("--task",   required=True, choices=["order", "interval", "simultaneity"])
    p.add_argument("--meta",   default="data/videos/metadata.csv")
    p.add_argument("--fps",    type=int, default=None, help="filter by fps (E004)")
    p.add_argument("--out",    default=None, help="override output csv path")
    p.add_argument("--limit",  type=int, default=None, help="videos per interval for quick runs")
    args = p.parse_args()

    fps_tag = f"_fps{args.fps}" if args.fps else ""
    out = Path(args.out) if args.out else Path(f"results/{args.model}_{args.task}{fps_tag}.csv")
    run(args.model, Path(args.meta), args.task, args.fps, out, args.limit)
