"""
Exp F — TempCompass yes/no (order + direction) under three visual conditions.

  real   : original TempCompass clip
  static : the clip's middle frame repeated for the clip's duration (no temporal evidence,
           same scene content)  -> isolates what a single frame + language prior can answer
  blank  : uniform gray clip (no visual evidence at all) -> pure question-conditioned prior

Labels are balanced (439 yes / 437 no over order+direction), so SDT d' and criterion c are
well defined on real video, and P(yes) on static/blank measures the prior.

Resumable: rows are appended per item; rerun skips (condition, video_id, question) already done.

Usage: python tc_yesno.py --model molmo2 --out results/molmo2_tc_yesno.csv
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models import load_model
import re

ITEMS = Path("data/tempcompass/yes_no_order_direction.csv")
DIRS = {
    "real": Path("data/tempcompass/videos"),
    "static": Path("data/tempcompass/videos_static"),
    "blank": Path("data/tempcompass/videos_blank"),
}
SUFFIX = "\nAnswer YES or NO."


def _parse_yn(text: str):
    """First YES/NO word decides (greedy first token is the answer; some adapters keep generating
    after EOS, e.g. Qwen2.5-VL-7B emits 'NO\n addCriterion ... YES')."""
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return None if m is None else int(m.group(1) == "YES")


FIELDS = ["condition", "video_id", "dim", "question", "gold", "pred", "correct", "response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", nargs="+", default=["real", "static", "blank"])
    ap.add_argument("--limit", type=int, default=0, help="debug: items per condition")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = pd.read_csv(ITEMS, dtype={"video_id": str})
    if args.limit:
        items = items.head(args.limit)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        prev = pd.read_csv(out, dtype={"video_id": str})
        done = set(zip(prev["condition"], prev["video_id"], prev["question"]))

    model = load_model(args.model)
    new_file = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for cond in args.conditions:
            todo = [r for _, r in items.iterrows()
                    if (cond, r["video_id"], r["question"]) not in done]
            for r in tqdm(todo, desc=f"{args.model}/{cond}"):
                vid = DIRS[cond] / f"{r['video_id']}.mp4"
                if not vid.exists():
                    continue
                try:
                    resp = model.ask(vid, str(r["question"]) + SUFFIX)
                except Exception as e:  # keep going; record failure as unparseable
                    resp = f"ERROR: {type(e).__name__}: {e}"
                pred = _parse_yn(resp)
                gold = 1 if str(r["answer"]).strip().lower() == "yes" else 0
                w.writerow({"condition": cond, "video_id": r["video_id"], "dim": r["dim"],
                            "question": r["question"], "gold": gold,
                            "pred": "" if pred is None else pred,
                            "correct": int(pred == gold) if pred is not None else 0,
                            "response": resp[:200].replace("\n", " ")})
                fh.flush()


if __name__ == "__main__":
    main()
