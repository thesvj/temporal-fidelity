"""
Exp H — TempCompass yes/no with first-token YES-NO margins (real clips only).

Same items and parser as tc_yesno.py; additionally records the first generated token's
log-probability margin between YES and NO, so the answer-vs-logits analysis of the controlled
task can be repeated on naturalistic video.
"""
import argparse, csv, math, re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import transformers.generation.utils as gu
CAP = {"first": None}
_orig = gu.GenerationMixin.generate
def _cap(self, *a, **k):
    want = k.get("return_dict_in_generate", False)
    k2 = dict(k); k2.update(output_scores=True, output_logits=True, return_dict_in_generate=True, max_new_tokens=8)
    try:
        out = _orig(self, *a, **k2)
    except Exception:
        CAP["first"] = None; k["max_new_tokens"] = 8; return _orig(self, *a, **k)
    lg = getattr(out, "logits", None) or getattr(out, "scores", None)
    CAP["first"] = lg[0][0].detach().float().cpu() if lg else None
    return out if want else out.sequences
gu.GenerationMixin.generate = _cap

from models import load_model

ITEMS = Path("data/tempcompass/yes_no_order_direction.csv")
VID = Path("data/tempcompass/videos")
SUFFIX = "\nAnswer YES or NO."
FIELDS = ["video_id", "dim", "question", "gold", "pred", "margin", "top_token", "response"]


def parse_yn(t):
    m = re.search(r"\b(YES|NO)\b", t.upper())
    return None if m is None else int(m.group(1) == "YES")


def find_tok(model):
    for v in vars(model).values():
        for c in (v, getattr(v, "tokenizer", None)):
            if c is not None and hasattr(c, "convert_tokens_to_ids") and hasattr(c, "encode"):
                return c
    return None


def ids_for(tok, words):
    s = set()
    for w in words:
        for x in (w, " " + w):
            e = tok.encode(x, add_special_tokens=False)
            if e: s.add(e[0])
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = pd.read_csv(ITEMS, dtype={"video_id": str})
    out = Path(args.out); done = set()
    if out.exists():
        done = set(zip(pd.read_csv(out, dtype={"video_id": str})["video_id"], pd.read_csv(out)["question"]))
    model = load_model(args.model)
    tok = find_tok(model)
    yes, no = ids_for(tok, ["YES","Yes","yes"]), ids_for(tok, ["NO","No","no"])
    print("yes", yes, "no", no, flush=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        for _, r in tqdm(items.iterrows(), total=len(items), desc=args.model):
            if (r["video_id"], r["question"]) in done: continue
            p = VID / f"{r['video_id']}.mp4"
            if not p.exists(): continue
            CAP["first"] = None
            try: resp = model.ask(p, str(r["question"]) + SUFFIX)
            except Exception as e: resp = f"ERROR: {type(e).__name__}: {e}"
            mg, top = float("nan"), ""
            L = CAP["first"]
            if L is not None and yes and no:
                lp = torch.log_softmax(L, -1)
                mg = float(torch.logsumexp(lp[yes], 0) - torch.logsumexp(lp[no], 0))
                top = tok.decode([int(L.argmax())]).strip()[:12]
            pr = parse_yn(resp)
            w.writerow({"video_id": r["video_id"], "dim": r["dim"], "question": r["question"],
                        "gold": 1 if str(r["answer"]).strip().lower() == "yes" else 0,
                        "pred": "" if pr is None else pr,
                        "margin": "" if math.isnan(mg) else round(mg, 4), "top_token": top,
                        "response": resp[:120].replace("\n", " ")})
            fh.flush()


if __name__ == "__main__":
    main()
