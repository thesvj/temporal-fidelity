"""Harness validation: TempCompass multi-choice with the official prompt and matching rules.

Each adapter is run at the frame count of a published score and at the 8 frames our harness uses, on the same
1,580 items, so the table can show: published / ours at the published setting / ours at the harness setting.
Prompt suffix and matching follow the TempCompass repository: "\\nPlease directly give the best option:";
a response counts if it equals the full option, is a bare letter, or starts with "A." / "A)" (same for B-D);
anything else is wrong (no LLM judge). The match rate is reported.

Resumable (appends per item). Usage:
  python tc_official.py --model qwen2.5-vl --frames 64 --out results_tc_official/qwen2.5-vl_f64.csv
"""
import argparse
import csv
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from models import load_model

VID = Path("data/tempcompass/videos")
SUFFIX = "\nPlease directly give the best option:"


def match(resp, answer):
    """(matched by rule, correct). Official order: exact, bare letter, 'A.' prefix, 'A)' prefix."""
    p = resp.strip()
    if p == answer:
        return True, True
    if p in ("A", "B", "C", "D"):
        return True, p == answer[0]
    for suf in (".", ")"):
        if any(p.startswith(L + suf) for L in "ABCD"):
            return True, p[0] == answer[0]
    return False, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=0, help="0 = the adapter's own sampling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv("data/tempcompass/multi_choice.csv")
    if a.limit:
        df = df.sample(a.limit, random_state=0)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        done = {(r["video_id"], r["question"]) for r in csv.DictReader(open(out)) if not r["response"].startswith("ERROR")}
    model = load_model(a.model)
    if a.frames:
        model.n = a.frames
    new = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["video_id", "dim", "question", "answer", "response", "matched", "correct"])
        if new:
            w.writeheader()
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"{a.model}/f{a.frames}"):
            if (str(r["video_id"]), str(r["question"])) in done:
                continue
            try:
                resp = model.ask(VID / f"{r['video_id']}.mp4", str(r["question"]) + SUFFIX)
            except Exception as e:
                resp = f"ERROR: {type(e).__name__}: {e}"
                del e
                gc.collect(); torch.cuda.empty_cache()      # an OOM otherwise leaves the allocator full for every later item
            m, c = match(resp, str(r["answer"]).strip())
            w.writerow(dict(video_id=r["video_id"], dim=r["dim"], question=r["question"], answer=r["answer"],
                            response=resp[:200], matched=int(m), correct=int(c)))
            fh.flush()
    res = pd.read_csv(out)
    print(f"TC_OFFICIAL {a.model} frames={a.frames} n={len(res)} acc={100 * res.correct.mean():.2f} "
          f"matched={100 * res.matched.mean():.1f}% errors={int(res.response.astype(str).str.startswith('ERROR').sum())} "
          + " ".join(f"{d}={100 * g.correct.mean():.1f}" for d, g in res.groupby("dim")))


if __name__ == "__main__":
    main()
