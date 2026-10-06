"""
Exp G3 — every controlled measurement on the v3 stimuli, in one pass.

Supersedes exp_g2.py (order/move/still/blank on v2) and exp_h.py (interval/prompt variants on v2).
Running them together matters: v2 drew the twin conditions from `rows[:240]`, which is the 67/100/200 ms
trials only, so the move question, the polarity control and the 32-frame comparison were all measured on
the three shortest intervals. v3 stratifies every subset across all 12 intervals.

Item sets (1,200 unique v3 videos; the question names one shape per trial, `name_first`):
  order_real      "did the {first-named} move before the {second-named}?"      gt = metadata gt
  order_32f       same question, 32 frames instead of 8, stratified 240        gt = metadata gt
  order_rev       polarity reversed ("... move AFTER ...")                      gt = 1 - gt
  order_forced    forced choice, "which moved first?"                           gt = metadata gt
  interval        4-way duration bin                                            gt = bin(interval_ms)
  moved_real      "did the {queried} move?" on the real clip (both move)        gt = 1
  moved_twin      same question on the frozen twin                              gt = 0
  moved_solo_hit  clip in which ONLY the queried shape moves                    gt = 1
  moved_solo_fa   clip in which only the OTHER shape moves                      gt = 0
  still_real      "did both shapes stay still?" on the real clip                gt = 0
  still_twin      same on the frozen twin                                       gt = 1
  order_twin      order question on the frozen twin                             gt = -1
  order_blank     order question on blank video                                 gt = -1

`moved_solo_fa` is the condition v2 lacked. Against a frozen twin, any change detector scores a high AUC
on the move question without binding the question to an object; separating moved_solo_hit from
moved_solo_fa requires that binding, because both clips contain exactly one moving shape.

Resumable (append + skip). Usage: python exp_g3.py --model molmo2 --out results/molmo2_expg3.csv
"""
import argparse
import csv
import math
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import transformers.generation.utils as gu

CAP = {"first": None}
_orig_generate = gu.GenerationMixin.generate


def _capturing_generate(self, *args, **kwargs):
    wanted_dict = kwargs.get("return_dict_in_generate", False)
    k2 = dict(kwargs)
    k2.update(output_scores=True, output_logits=True, return_dict_in_generate=True, max_new_tokens=8)
    try:
        out = _orig_generate(self, *args, **k2)
    except Exception:
        CAP["first"] = None
        kwargs["max_new_tokens"] = 8
        return _orig_generate(self, *args, **kwargs)
    logits = getattr(out, "logits", None) or getattr(out, "scores", None)
    CAP["first"] = logits[0][0].detach().float().cpu() if logits else None
    return out if wanted_dict else out.sequences


gu.GenerationMixin.generate = _capturing_generate

from models import load_model  # noqa: E402  (import after patch)

# {n1}, {n2}: how the question names the two objects. "red square" on the shape stimuli (colour + form),
# a noun such as "weightlifter" on the real-footage stimuli (generate_real.py). Same sentences for both.
P_ORDER = "In this video, did the {n1} move BEFORE the {n2}? Answer YES or NO."
P_REV = "In this video, did the {n2} move AFTER the {n1}? Answer YES or NO."
P_FORCED = ("Which shape moved first in this video, the {n1} or the {n2}? "
            "Answer with A or B only.")
P_MOVED = "In this video, did the {n1} move? Answer YES or NO."
# Real-footage stimuli only: does the model attach each noun to the right panel? Asked on the frozen twin,
# which contains no event, so the answer cannot come from the order.
P_BIND = "In this video, is the {n1} in the top half of the frame? Answer YES or NO."
P_STILL = "In this video, did both shapes stay still the whole time? Answer YES or NO."
P_INTERVAL = ("How much time passed between the two events? Answer exactly one letter: "
              "(A) Less than 1 second, (B) 1-2 seconds, (C) 2-5 seconds, (D) More than 5 seconds.")
BINS = [(0, 1000, "A"), (1000, 2000, "B"), (2000, 5000, "C"), (5000, 1e9, "D")]
FIELDS = ["item", "path", "interval_ms", "gt", "pred", "margin", "lpA", "lpB", "lpC", "lpD",
          "n_frames", "queried", "top_token", "response"]
YESNO = {"order_real", "order_32f", "order_rev", "moved_real", "moved_twin",
         "moved_solo_hit", "moved_solo_fa", "still_real", "still_twin", "order_twin", "order_blank",
         "bind_twin"}


def bin_of(ms):
    for lo, hi, letter in BINS:
        if lo <= ms < hi:
            return letter
    return "D"


def parse_yn(text):
    """First YES/NO word wins: some adapters keep generating past the answer."""
    m = re.search(r"\b(YES|NO)\b", text.upper())
    return None if m is None else int(m.group(1) == "YES")


def parse_letter(text, allowed="ABCD"):
    t = text.upper()
    m = re.search(rf"\(([{allowed}])\)", t) or re.search(rf"\bANSWER\s*(?:IS|:)\s*([{allowed}])\b", t)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{allowed}])\b", t)
    return m.group(1) if m else None


def parse_forced(text, names):
    """Letter first; models often answer by naming the shape, and often pair the right colour with the
    wrong form, so the colour (unique within a trial) is the reliable identifier."""
    letter = parse_letter(text, "AB")
    if letter is not None:
        return int(letter == "A")
    t = text.upper()
    a = re.search(rf"\b{names['c1'].upper()}\b", t)
    b = re.search(rf"\b{names['c2'].upper()}\b", t)
    if a and not b:
        return 1
    if b and not a:
        return 0
    if a and b:
        return int(a.start() < b.start())
    return None


def find_tokenizer(model):
    for v in vars(model).values():
        for cand in (v, getattr(v, "tokenizer", None)):
            if cand is not None and hasattr(cand, "convert_tokens_to_ids") and hasattr(cand, "encode"):
                return cand
    return None


def variant_ids(tok, words, prefixes=("", " ")):
    """First-token ids per surface form. "(" is only safe for the letter options: for YES/NO some
    tokenizers emit "(" as its own token, which would then sit on both sides of the margin."""
    ids = set()
    for w in words:
        for p in prefixes:
            enc = tok.encode(p + w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def naming(r):
    """The question names one shape first; `queried` is the shape the move question asks about."""
    if "noun_a" in r:   # real footage: one noun per panel
        a, b = (dict(c=r["noun_a"], s=""), dict(c=r["noun_b"], s=""))
    else:
        a, b = (dict(c=r["color_a"], s=r["shape_a"]), dict(c=r["color_b"], s=r["shape_b"]))
    first, second, q = (a, b, "a") if r["name_first"] == "a" else (b, a, "b")
    return dict(c1=first["c"], s1=first["s"], c2=second["c"], s2=second["s"],
                n1=f"{first['c']} {first['s']}".strip(), n2=f"{second['c']} {second['s']}".strip()), q


def build_items_real(meta):
    """Real-footage stimuli: the order question on every trial (each flipped pair is two trials); the
    paraphrase, the move question and the order question on the frozen twin for the trials that have a
    twin. Forced choice, interval, 32-frame and still items are not run on these stimuli, and there
    are no one-panel-moves clips."""
    items = []
    for _, r in meta.iterrows():
        names, q = naming(r)
        gt = int(r["gt"])
        items.append(("order_real", r, 8, gt, Path(r["path"]), names, q))
        if isinstance(r.get("twin_path"), str):
            tw = Path(r["twin_path"])
            items += [("order_rev", r, 8, gt, Path(r["path"]), names, q),
                      ("moved_real", r, 8, 1, Path(r["path"]), names, q),
                      ("moved_twin", r, 8, 0, tw, names, q),
                      ("order_twin", r, 8, -1, tw, names, q),
                      # panel a is the top panel, so the first-named noun is on top iff name_first == "a"
                      ("bind_twin", r, 8, int(r["name_first"] == "a"), tw, names, q)]
    return items


def build_items(meta):
    if "noun_a" in meta.columns:
        return build_items_real(meta)
    items = []
    for _, r in meta.iterrows():
        names, q = naming(r)
        gt = int(r["gt"])   # r.gt is pandas Series.gt (greater-than), not the column
        items += [("order_real", r, 8, gt, Path(r["path"]), names, q),
                  # "did N2 move AFTER N1?" is TRUE exactly when "did N1 move BEFORE N2?" is true, so the
                  # label is unchanged. This is a paraphrase control, not a negation: it swaps which shape
                  # is mentioned first while holding the truth value, so a model that answers "is the
                  # first-mentioned shape the first mover" flips its answer and scores below chance.
                  ("order_rev", r, 8, gt, Path(r["path"]), names, q),
                  ("order_forced", r, 8, gt, Path(r["path"]), names, q),
                  ("interval", r, 8, bin_of(int(r["interval_ms"])), Path(r["path"]), names, q)]
        if isinstance(getattr(r, "twin_path", None), str):
            tw = Path(r["twin_path"])
            items += [("order_32f", r, 32, gt, Path(r["path"]), names, q),
                      ("moved_real", r, 8, 1, Path(r["path"]), names, q),
                      ("moved_twin", r, 8, 0, tw, names, q),
                      ("still_real", r, 8, 0, Path(r["path"]), names, q),
                      ("still_twin", r, 8, 1, tw, names, q),
                      ("order_twin", r, 8, -1, tw, names, q)]
        hit = getattr(r, f"solo_{q}_path", None)
        fa = getattr(r, "solo_b_path" if q == "a" else "solo_a_path", None)
        if isinstance(hit, str) and isinstance(fa, str):
            items += [("moved_solo_hit", r, 8, 1, Path(hit), names, q),
                      ("moved_solo_fa", r, 8, 0, Path(fa), names, q)]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="data/videos_v3/metadata.csv")
    ap.add_argument("--blank", default="data/blank_videos/metadata.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    items = build_items(meta)
    if Path(args.blank).exists():
        blank = pd.read_csv(args.blank)
        blank = blank[blank.condition == "blank"] if "condition" in blank.columns else blank
        real = None
        if "noun_a" in meta.columns:   # every noun pair of the run, in the order the question names them
            real = sorted({(r["noun_a"], r["noun_b"]) if r["name_first"] == "a" else (r["noun_b"], r["noun_a"])
                           for _, r in meta.iterrows()})
        for i, (_, r) in enumerate(blank.iterrows()):
            names = dict(c1="red", s1="square", c2="blue", s2="circle", n1="red square", n2="blue circle")
            if real:   # the nouns of the real-footage trials, so the blank baseline shares their wording
                n1, n2 = real[i % len(real)]
                names = dict(c1=n1, s1="", c2=n2, s2="", n1=n1, n2=n2)
            items.append(("order_blank", r, 8, -1, Path(r["path"]), names, "a"))
    only = os.environ.get("EXPG3_ONLY")
    if only:   # run a subset of the item kinds, e.g. EXPG3_ONLY=order_real for the frame-alignment arms
        keep = set(only.split(","))
        items = [it for it in items if it[0] in keep]
    if args.limit:
        items = items[: args.limit]

    out = Path(args.out)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        # everything as text: letting pandas infer types turns pred into 1.0/0.0 once a blank is present,
        # and the analysis compares it with "1"/"0"
        prev = pd.read_csv(out, dtype=str, keep_default_na=False)
        # a row whose response is an exception is not done: drop it and run the item again
        ok = ~prev["response"].str.startswith("ERROR:")
        if not ok.all():
            tmp = out.with_suffix(".tmp")
            prev[ok].to_csv(tmp, index=False)
            tmp.replace(out)
            print(f"dropped {int((~ok).sum())} ERROR rows from {out}; they will be rerun", flush=True)
        done = set(zip(prev[ok]["item"], prev[ok]["path"]))

    model = load_model(args.model)
    tok = find_tokenizer(model)
    yes_ids = variant_ids(tok, ["YES", "Yes", "yes"]) if tok else []
    no_ids = variant_ids(tok, ["NO", "No", "no"]) if tok else []
    shared = set(yes_ids) & set(no_ids)
    yes_ids = [i for i in yes_ids if i not in shared]
    no_ids = [i for i in no_ids if i not in shared]
    letter_ids = {L: variant_ids(tok, [L], ("", " ", "(")) for L in "ABCD"} if tok else {}
    seen = [i for L in "ABCD" for i in letter_ids.get(L, [])]
    dup = {i for i in seen if seen.count(i) > 1}
    letter_ids = {L: [i for i in v if i not in dup] for L, v in letter_ids.items()}
    print(f"tokenizer={type(tok).__name__} yes={yes_ids} no={no_ids} letters={letter_ids}", flush=True)

    # adapters whose ask() reads self.n (molmo2 / video-llama2 fix their frame count at load time)
    default_n = getattr(model, "n", None) if args.model not in ("molmo2", "video-llama2") else None

    new = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for kind, r, nf, gt, path, names, q in tqdm(items, desc=args.model):
            if default_n is None and kind == "order_32f":
                continue
            if (kind, str(path)) in done:
                continue
            if default_n is not None:
                model.n = nf
            if kind in ("order_real", "order_32f", "order_twin", "order_blank"):
                prompt = P_ORDER.format(**names)
            elif kind == "order_rev":
                prompt = P_REV.format(**names)
            elif kind == "order_forced":
                prompt = P_FORCED.format(**names)
            elif kind == "interval":
                prompt = P_INTERVAL
            elif kind == "bind_twin":
                prompt = P_BIND.format(**names)
            elif kind.startswith("still"):
                prompt = P_STILL
            else:
                prompt = P_MOVED.format(**names)

            CAP["first"] = None
            try:
                resp = model.ask(path, prompt)
            except Exception as e:
                resp = f"ERROR: {type(e).__name__}: {e}"

            row = {f: "" for f in FIELDS}
            L = CAP["first"]
            lp = torch.log_softmax(L, -1) if L is not None else None
            if lp is not None:
                for letter in "ABCD":
                    ids = letter_ids.get(letter) or []
                    if ids:
                        row[f"lp{letter}"] = round(float(torch.logsumexp(lp[ids], 0)), 4)
                row["top_token"] = (tok.decode([int(L.argmax())]) if tok else "").strip()[:12]

            margin = float("nan")
            if kind in YESNO:
                pred = parse_yn(resp)
                if lp is not None and yes_ids and no_ids:
                    margin = float(torch.logsumexp(lp[yes_ids], 0) - torch.logsumexp(lp[no_ids], 0))
            elif kind == "order_forced":
                pred = parse_forced(resp, names)
                if row["lpA"] != "" and row["lpB"] != "":
                    margin = row["lpA"] - row["lpB"]
            else:
                pred = parse_letter(resp, "ABCD")

            row.update(item=kind, path=str(path), interval_ms=r.get("interval_ms", 0), gt=gt,
                       pred="" if pred is None else pred,
                       margin="" if math.isnan(margin) else round(margin, 4),
                       n_frames=nf if default_n is not None else "default", queried=q,
                       response=resp[:120].replace("\n", " "))
            w.writerow(row)
            fh.flush()
    if default_n is not None:
        model.n = default_n


if __name__ == "__main__":
    main()
