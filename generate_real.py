"""Stimulus generator for the real-footage run: two real bursts with a controlled offset.

Same trial structure as generate_v3.py, with the two shapes replaced by two panels of real footage.
Each panel shows one *burst*: K+1 consecutive frames of a source video (static camera, one person).
The panel holds frame 0 until its onset, plays frames 1..K, then holds frame K to the end. So each
panel changes state exactly once, like a shape that jumps, and the event is 0.5 s of real motion.

Why a bounded burst and not "start playing and keep playing": if the footage kept running, the panel
that started first would be further along in every later frame, and the order could be read off the
last frame alone at any interval. With a bounded burst the two orders of a trial show identical pixels
on every frame outside [f_first, f_second + K - 1), which verify_real.py checks bit for bit.

Every trial is rendered in both orders (a flipped pair), each a trial of its own. The question names
the same panel first in both, so the label flips with the order and is exactly balanced against every
property of the footage, the layout and the timing. Pairs are drawn across activity categories, and
each burst is used equally often on top and at the bottom.

A stratified subset also gets a frozen twin (both panels hold frame 0 throughout).

Usage:
  python generate_real.py --clips clips.csv --src data/real_src/clips --out data/videos_real \
      --n 50 --seed 0 --twins data/twin_real --n-twins 240
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from generate_v3 import CODEC, EXT, FPS, INTERVALS_MS

W = H = 480
PH = H // 2          # panel height; panels are W x PH, stacked
K = 15               # played frames per burst


def load_burst(src: Path, r: dict):
    """K+1 panels (PH x W x 3, uint8) for one row of clips.csv."""
    cap = cv2.VideoCapture(str(src / r["file"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["start"]))
    l, t, w, h = (float(v) for v in r["crop"].split())
    out = []
    for _ in range(K + 1):
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"{r['file']}: ran out of frames at burst start {r['start']}")
        fh, fw = f.shape[:2]
        x, y = int(round(l * fw)), int(round(t * fh))
        out.append(cv2.resize(f[y:y + int(round(h * fh)), x:x + int(round(w * fw))], (W, PH),
                              interpolation=cv2.INTER_AREA))
    cap.release()
    # A frame-rate conversion can duplicate a frame. At either end of the burst that would make the panel
    # leave its first still one frame late, or reach its last still one frame early, and the visibility
    # window would be off by a frame for every trial using the burst.
    if np.array_equal(out[0], out[1]) or np.array_equal(out[K - 1], out[K]):
        raise RuntimeError(f"{r['burst']}: duplicated frame at a burst end; shift its start")
    return out


def sample_trials(bursts, n, rng):
    """n base trials per interval. Every burst appears equally often as the top and as the bottom
    panel (up to rounding); the two panels of a trial come from different categories and sources."""
    ids = list(bursts)
    total = n * len(INTERVALS_MS)

    def ok(a, b):
        return (bursts[a]["category"] != bursts[b]["category"] and bursts[a]["noun"] != bursts[b]["noun"]
                and bursts[a]["file"] != bursts[b]["file"])

    tops = [ids[i % len(ids)] for i in range(total)]
    bots = tops[:]
    rng.shuffle(tops); rng.shuffle(bots)
    for _ in range(200 * total):          # repair by swapping bottoms; both swapped positions must end up valid
        bad = [i for i in range(total) if not ok(tops[i], bots[i])]
        if not bad:
            break
        i, j = rng.choice(bad), rng.randrange(total)
        if ok(tops[i], bots[j]) and ok(tops[j], bots[i]):
            bots[i], bots[j] = bots[j], bots[i]
    else:
        raise RuntimeError("could not pair bursts across categories; too few categories?")
    trials = []
    for k, (a, b) in enumerate(zip(tops, bots)):
        trials.append(dict(
            pair_id=k, interval_ms=INTERVALS_MS[k % len(INTERVALS_MS)],
            burst_a=a, burst_b=b,                      # a = top panel, b = bottom panel
            noun_a=bursts[a]["noun"], noun_b=bursts[b]["noun"],
            cat_a=bursts[a]["category"], cat_b=bursts[b]["category"],
            src_a=bursts[a]["file"], src_b=bursts[b]["file"],
            who_a=bursts[a]["uploader"], who_b=bursts[b]["uploader"],   # resampling unit: one person films many clips
            a_first=rng.random() < 0.5,                # order of the unflipped member
            name_first=rng.choice("ab"),               # same for both members of the pair
            lead_s=round(rng.uniform(0.7, 1.3), 2),
            tail_s=round(rng.uniform(1.2, 1.8), 2),
        ))
    trials.sort(key=lambda t: (t["interval_ms"], t["pair_id"]))
    return trials


def render(path: Path, t: dict, frames, frozen: bool = False, flip: bool = False):
    lead, tail, dt = t["lead_s"], t["tail_s"], t["interval_ms"] / 1000
    f_first = int(lead * FPS)
    gap = int(round(dt * FPS))
    f_second = f_first + gap
    n_frames = f_second + K + int(tail * FPS)          # the tail starts when the second burst ends
    a_first = (not t["a_first"]) if flip else t["a_first"]
    fa, fb = (f_first, f_second) if a_first else (f_second, f_first)
    A, B = frames[t["burst_a"]], frames[t["burst_b"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*CODEC), FPS, (W, H))   # FFV1: see generate_v3.render
    for i in range(n_frames):
        # frame fa is the first one that differs from the still, as a shape has moved at its onset frame
        pa = 0 if frozen else min(max(i - fa + 1, 0), K)
        pb = 0 if frozen else min(max(i - fb + 1, 0), K)
        vw.write(np.vstack([A[pa], B[pb]]))
    vw.release()
    named_first_moved_first = a_first if t["name_first"] == "a" else (not a_first)
    return dict(t, path=str(path), a_first=a_first, flipped=int(flip), n_frames=n_frames, gap_frames=gap,
                f_first=f_first, f_second=f_second, burst_frames=K, fps=FPS, gt=int(named_first_moved_first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips.csv")
    ap.add_argument("--src", default="data/real_src/clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50, help="base trials per interval; each is rendered in both orders")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twins", default=None)
    ap.add_argument("--n-twins", type=int, default=240, help="total twins, spread evenly over intervals")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bursts = {r["burst"]: r for r in csv.DictReader(open(args.clips))}
    frames = {b: load_burst(Path(args.src), r) for b, r in bursts.items()}
    trials = sample_trials(bursts, args.n, rng)
    per_interval_twins = max(1, args.n_twins // len(INTERVALS_MS))
    out, rows, seen = Path(args.out), [], {}
    for t in trials:
        ms, k = t["interval_ms"], t["pair_id"]
        j = seen[ms] = seen.get(ms, -1) + 1
        r0 = render(out / f"int_{ms}ms" / f"real_{ms}_{k:05d}{EXT}", t, frames)
        r1 = render(out / f"int_{ms}ms" / f"real_{ms}_{k:05d}_flip{EXT}", t, frames, flip=True)
        if args.twins and j < per_interval_twins:
            tw = render(Path(args.twins) / Path(r0["path"]).name, t, frames, frozen=True)
            r0["twin_path"] = tw["path"]
        rows += [r0, r1]
    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({kk for r in rows for kk in r}))
        w.writeheader()
        w.writerows(rows)
    n_tw = sum("twin_path" in r for r in rows)
    print(f"rendered {len(rows)} trials ({len(trials)} flipped pairs, +{n_tw} twins) from {len(bursts)} bursts -> {out}")


if __name__ == "__main__":
    main()
