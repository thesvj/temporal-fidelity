"""Checks the real-footage stimulus set has the properties the paper claims for it. CPU only.

Every check prints PASS/FAIL and the number it checked.

  1 pairs            every trial has its order-flipped partner; the two have opposite labels and share
                     every other field, so the label is exactly balanced against footage, layout, timing
  2 encoding leak    in a flipped pair every decoded frame outside [f_first, f_second + K) is
                     bit-identical (K - 1 for the last played frame), so nothing outside the two events separates the orders
  3 twins            every frame of a frozen twin equals frame 0 of its real clip
  4 burst quality    each panel has changed one frame after its onset, and its pre and post stills differ
  5 ideal observer   reading only the frames a sampler delivers, an observer that compares each panel
                     with its first and last state gets the order right on every trial the visibility
                     rule calls visible and has no evidence on the rest
  6 recorded frames  with --res: every (sampler, clip) has logged frame indices, so visibility comes from
                     the loader each model really runs and never from our reading of it

It also writes <meta dir>/evidence.csv: per trial and sampler, the class of the trial (state: a delivered
frame shows one panel changed and the other not; phase: the delivered frames differ between the two
orders only while both panels are mid-burst; blind: the delivered frames are identical for both orders)
and the evidence strength (largest mean absolute difference, over the delivered frames, between the
trial and its order-flipped partner after resizing to 224 px).

Usage: python verify_real.py --meta data/videos_real/metadata.csv [--max-pairs 0]
"""
import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np

import analyze_v3 as av

RESULTS = []
ONSET_MIN, STILL_MIN = 0.4, 1.5      # mean abs grey-level change over the panel: after 1 frame; pre vs post
SAME = ["interval_ms", "burst_a", "burst_b", "noun_a", "noun_b", "name_first", "lead_s", "tail_s",
        "f_first", "f_second", "n_frames", "burst_frames"]


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return np.array(out)


def observer(v, idx):
    """+1: top panel first, -1: bottom first, 0: no evidence. A panel is `pre` if it equals its state in
    the first frame of the clip and `post` if it equals its state in the last frame."""
    h = v.shape[1] // 2
    for i in idx:
        top, bot = v[i, :h], v[i, h:]
        pre = [np.array_equal(top, v[0, :h]), np.array_equal(bot, v[0, h:])]
        post = [np.array_equal(top, v[-1, :h]), np.array_equal(bot, v[-1, h:])]
        if pre[0] != pre[1]:
            return 1 if pre[1] else -1
        if post[0] != post[1]:
            return 1 if post[0] else -1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="pixel checks on this many pairs (0 = all)")
    ap.add_argument("--res", default=None, help="results dir with sampled_frames_*.jsonl, if recorded")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.meta)))
    if args.res:
        av.load_recorded(args.res)
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair_id"]].append(r)

    bad = [k for k, p in pairs.items() if len(p) != 2 or p[0]["gt"] == p[1]["gt"] or p[0]["a_first"] == p[1]["a_first"]
           or any(p[0][c] != p[1][c] for c in SAME)]
    p_gt = np.mean([int(r["gt"]) for r in rows])
    check("pairs", not bad, f"{len(pairs)} flipped pairs, {len(rows)} trials, P(gt=1) = {p_gt:.3f}, {len(bad)} malformed")
    same_cat = sum(r["cat_a"] == r["cat_b"] or r["src_a"] == r["src_b"] for r in rows)
    check("cross-category", same_cat == 0, f"{same_cat} trials pair two bursts of one category or source")

    todo = list(pairs.values())[: args.max_pairs or None]
    leak = weak = 0
    ev_rows, classes = [], defaultdict(lambda: defaultdict(int))
    obs = defaultdict(lambda: [0, 0, 0, 0])     # sampler -> visible right, visible total, invisible decided, invisible total
    samplers = {"uniform8": "qwen2.5-vl", "midpoint8": "video-llama2", "molmo2": "molmo2", "vcf64": "videochat-flash"}
    for p in todo:
        a, b = sorted(p, key=lambda r: int(r["flipped"]))
        va, vb = frames(a["path"]), frames(b["path"])
        f1, f2, K = int(a["f_first"]), int(a["f_second"]), int(a["burst_frames"])
        outside = [i for i in range(len(va)) if not f1 <= i < f2 + K - 1]
        if va.shape != vb.shape or not np.array_equal(va[outside], vb[outside]):
            leak += 1
        h = va.shape[1] // 2
        for v, r in ((va, a), (vb, b)):
            top_first = r["a_first"] == "True"
            on = {True: (f1, f2), False: (f2, f1)}[top_first]            # onset of top, bottom
            for sl, f in ((slice(0, h), on[0]), (slice(h, None), on[1])):
                d1 = np.abs(v[f, sl].astype(np.int16) - v[f - 1, sl]).mean()
                dK = np.abs(v[-1, sl].astype(np.int16) - v[0, sl]).mean()
                weak += d1 < ONSET_MIN or dK < STILL_MIN
            for name, model in samplers.items():
                idx = [i for i in av.sampled(r, model) if i < len(v)]
                dec = observer(v, idx)
                diff = [float(np.abs(cv2.resize(va[i], (224, 224), interpolation=cv2.INTER_AREA).astype(np.int16)
                                     - cv2.resize(vb[i], (224, 224), interpolation=cv2.INTER_AREA)).mean()) for i in idx]
                same = all(np.array_equal(va[i], vb[i]) for i in idx)
                cls = "state" if av.visible(r, model) else "blind" if same else "phase"
                classes[name][cls] += 1
                ev_rows.append(dict(path=r["path"], sampler=name, cls=cls, evidence=round(max(diff), 3)))
                o = obs[name]
                if av.visible(r, model):
                    o[1] += 1; o[0] += dec == (1 if top_first else -1)
                else:
                    o[3] += 1; o[2] += dec != 0
    with open(str(args.meta).rsplit("/", 1)[0] + "/evidence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "sampler", "cls", "evidence"]); w.writeheader(); w.writerows(ev_rows)
    for name, c in classes.items():
        print(f"       trial classes [{name}]: state {c['state']}, phase-only {c['phase']}, blind {c['blind']}")
    if args.res:
        missing = sum((model, r["path"]) not in av.RECORDED for r in rows for model in samplers.values())
        check("recorded frames", missing == 0, f"{len(rows) * len(samplers)} (sampler, clip) entries expected, {missing} missing")
    check("encoding leak", leak == 0, f"{len(todo)} pairs, {leak} differ outside [f_first, f_second + K - 1)")
    check("burst quality", weak == 0, f"{4 * len(todo)} panel events, {weak} below threshold "
                                      f"(change after 1 frame >= {ONSET_MIN}, pre vs post >= {STILL_MIN})")
    for name, (vr, vt, idec, it) in obs.items():
        check(f"ideal observer [{name}]", vr == vt and idec == 0,
              f"visible {vr}/{vt} right, invisible {idec}/{it} decided")

    tw = [r for r in rows if r.get("twin_path")][: args.max_pairs or None]
    bad_tw = 0
    for r in tw:
        v, t = frames(r["path"]), frames(r["twin_path"])
        bad_tw += len(v) != len(t) or not all(np.array_equal(f, v[0]) for f in t)
    check("twins", bad_tw == 0, f"{len(tw)} twins, {bad_tw} with a frame that is not frame 0 of the real clip")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    raise SystemExit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
