"""Stimulus generator v3 — removes the structural cue v2 left behind.

v1 randomized only the colour/form pairing and the order, so 1,200 trials were 72 distinct videos.
v2 randomized the cosmetic nuisances (colour, form, size, position, jump, lead-in, tail, background) but
kept three things that make the label readable without binding the question to an object:

  * shape A was always drawn in the top-left region and always moved up/left;
  * shape B was always bottom-right and always moved down/right;
  * the question always named A first.

So "did the {A} move before the {B}?" was answerable as "did the up-left motion come first?", and a probe
predicting `a_first` could read the direction of the first motion instead of the order. v3 fixes all three:

  * both shapes draw a direction from the same 8-way pool;
  * both start positions come from the same region, with rejection sampling for separation and for
    staying on canvas before and after the jump;
  * `name_first` is drawn per trial, so the shape the question names first is the first mover only half
    the time, and the label is "did the shape the question named first move first?".

It also renders, for a subset, a *matched pair*: the same trial with the order flipped and every other
parameter held fixed. Outside the interval between the two jumps the two videos depict identical pixels,
so any probe that separates them there is reading an encoding artefact rather than the order
(see verify_v3.py). Twins are stratified across all 12 intervals rather than taken from the head of the
file, which in v2 restricted every twin condition to 67/100/200 ms.

Usage:
  python generate_v3.py --out data/videos_v3 --n 100 --seed 0 --twins data/twin_v3 --pairs 240
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

W = H = 480
FPS = 30
INTERVALS_MS = [67, 100, 200, 333, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]
COLORS = {"red": (0, 0, 220), "blue": (220, 0, 0), "green": (0, 160, 0),
          "yellow": (0, 200, 220), "magenta": (200, 0, 200), "cyan": (200, 200, 0)}
FORMS = ["square", "circle", "triangle"]
CODEC, EXT = "FFV1", ".avi"   # lossless; see render()
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
MARGIN = 14          # keep this many px between a shape's bounding box and the canvas edge
MIN_SEP = 1.25       # min centre distance as a multiple of the two half-extents summed


def _draw(img, form, bgr, x, y, size):
    h = size // 2
    if form == "square":
        cv2.rectangle(img, (x - h, y - h), (x + h, y + h), bgr, -1)
    elif form == "circle":
        cv2.circle(img, (x, y), h, bgr, -1)
    else:
        pts = np.array([[x, y - h], [x - h, y + h], [x + h, y + h]], np.int32)
        cv2.fillPoly(img, [pts], bgr)


def _on_canvas(x, y, size):
    h = size // 2 + MARGIN
    return h <= x <= W - h and h <= y <= H - h


def _apart(xa, ya, sa, xb, yb, sb):
    need = MIN_SEP * (sa / 2 + sb / 2)
    return (xa - xb) ** 2 + (ya - yb) ** 2 >= need ** 2


def sample_trial(rng, interval_ms):
    """Every nuisance parameter drawn per trial, from pools shared by the two shapes."""
    ca, cb = rng.sample(list(COLORS), 2)
    for _ in range(4000):
        sa, sb = rng.randint(34, 50), rng.randint(34, 50)
        ax, ay = rng.randint(60, W - 60), rng.randint(60, H - 60)
        bx, by = rng.randint(60, W - 60), rng.randint(60, H - 60)
        da, db = rng.choice(DIRS), rng.choice(DIRS)
        ja, jb = rng.randint(45, 75), rng.randint(45, 75)
        ax2, ay2 = ax + da[0] * ja, ay + da[1] * ja
        bx2, by2 = bx + db[0] * jb, by + db[1] * jb
        if not (_on_canvas(ax, ay, sa) and _on_canvas(bx, by, sb)
                and _on_canvas(ax2, ay2, sa) and _on_canvas(bx2, by2, sb)):
            continue
        # keep them apart in all four configurations the video passes through
        if not all(_apart(px, py, sa, qx, qy, sb) for px, py in ((ax, ay), (ax2, ay2))
                   for qx, qy in ((bx, by), (bx2, by2))):
            continue
        return dict(
            interval_ms=interval_ms,
            a_first=rng.random() < 0.5,
            name_first=None,          # assigned by balance_labels() once all trials are drawn

            color_a=ca, color_b=cb,
            shape_a=rng.choice(FORMS), shape_b=rng.choice(FORMS),
            size_a=sa, size_b=sb,
            ax=ax, ay=ay, bx=bx, by=by,
            jump_a=ja, jump_b=jb, dir_a=da, dir_b=db,
            lead_s=round(rng.uniform(0.7, 1.3), 2),
            tail_s=round(rng.uniform(1.2, 1.8), 2),
            bg=rng.choice([205, 212, 220, 228]),
        )
    raise RuntimeError(f"rejection sampler failed at {interval_ms} ms")


def _cramers_v(labels, vals):
    rx, ry = sorted(set(labels)), sorted(set(vals))
    if len(rx) < 2 or len(ry) < 2:
        return 0.0
    tab = np.zeros((len(rx), len(ry)))
    for a, b in zip(labels, vals):
        tab[rx.index(a), ry.index(b)] += 1
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def balance_labels(rows, rng, tries=200):
    """Assign `name_first` so the behavioural label is balanced against the nuisances by construction.

    The label is "did the shape the question named first move first", i.e. a_first XOR name_first=='b'.
    `name_first` changes only the wording of the question, never a pixel, so it is free to assign after
    the videos are specified. Balancing it within each (dir_a, dir_b, a_first) cell makes the label
    independent of anything computable from those three -- including the direction of the FIRST mover,
    which is the cue v2 left in place.

    Drawing it at random instead leaves finite-sample associations: on one seed `dir_a` predicted the
    label at p = .009 uncorrected. So we also reject any assignment whose association with any nuisance
    exceeds what random labels produce (97.5th percentile of a simulated null), and redraw.
    """
    keys = ["color_a", "color_b", "shape_a", "shape_b", "size_a", "size_b", "ax", "ay", "bx", "by",
            "jump_a", "jump_b", "dir_a", "dir_b", "lead_s", "tail_s", "bg", "interval_ms"]

    def binned(vals, k=8):
        try:
            nums = [float(v) for v in vals]
        except (TypeError, ValueError):
            return [str(v) for v in vals]
        if len(set(nums)) <= k:
            return [str(v) for v in vals]
        qs = np.quantile(nums, np.linspace(0, 1, k + 1)[1:-1])
        return [str(int(np.searchsorted(qs, v))) for v in nums]

    cols = {k: binned([r[k] for r in rows]) for k in keys}
    first_dir = [str(r["dir_a"] if r["a_first"] else r["dir_b"]) for r in rows]
    cols["_first_mover_dir"] = first_dir

    # null: how large does the association get with labels assigned at random?
    thresh = {}
    for k, v in cols.items():
        sims = []
        for _ in range(200):
            lab = [str(rng.random() < 0.5) for _ in rows]
            sims.append(_cramers_v(lab, v))
        thresh[k] = float(np.percentile(sims, 97.5))

    cells = {}
    for i, r in enumerate(rows):
        cells.setdefault((str(r["dir_a"]), str(r["dir_b"]), bool(r["a_first"])), []).append(i)

    best = None
    for _ in range(tries):
        for idx in cells.values():
            order = idx[:]
            rng.shuffle(order)
            for j, i in enumerate(order):
                rows[i]["name_first"] = "a" if j % 2 == 0 else "b"
        lab = [str(bool(r["a_first"]) == (r["name_first"] == "a")) for r in rows]
        worst = max((_cramers_v(lab, v) / max(thresh[k], 1e-9), k) for k, v in cols.items())
        if best is None or worst[0] < best[0]:
            best = (worst[0], worst[1], [r["name_first"] for r in rows])
        if worst[0] <= 1.0:
            print(f"  labels balanced: largest association {worst[0]:.2f}x the random-label null ({worst[1]})")
            return
    for r, nf in zip(rows, best[2]):
        r["name_first"] = nf
    print(f"  labels balanced (best of {tries}): {best[0]:.2f}x the null ({best[1]})")


def render(path: Path, t: dict, frozen: bool = False, flip: bool = False, only: str = None):
    """only='a' renders a clip in which A jumps and B never moves (and vice versa). Asked "did the {A}
    move?", a change detector answers YES to both; only a model that binds the question to an object
    answers YES to only='a' and NO to only='b'. Without this condition the move-question AUC against a
    frozen twin is explained by any global motion signal."""
    lead, tail = t["lead_s"], t["tail_s"]
    dt = t["interval_ms"] / 1000
    n_frames = int((lead + dt + tail) * FPS)
    f_first = int(lead * FPS)
    gap = int(round(dt * FPS))
    f_second = f_first + gap
    a_first = (not t["a_first"]) if flip else t["a_first"]
    fa, fb = (f_first, f_second) if a_first else (f_second, f_first)
    ax, ay, bx, by = t["ax"], t["ay"], t["bx"], t["by"]
    path.parent.mkdir(parents=True, exist_ok=True)
    # FFV1, not mp4v. mp4v predicts each frame from its neighbours, so two clips that differ only in the
    # order of the two jumps decode to different pixels *outside* the interval between them (measured:
    # 50/889 frames, up to 38 grey levels). That is an order cue on trials the paper calls unanswerable,
    # and it is the most likely source of the probe reading .650 on invisible trials. FFV1 is lossless,
    # so identical inputs decode identically; decord reads it in every adapter venv (verify_v3.py:4).
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*CODEC), FPS, (W, H))
    moved_a = moved_b = False
    for i in range(n_frames):
        img = np.full((H, W, 3), t["bg"], np.uint8)
        a_moves = not frozen and only != "b"
        b_moves = not frozen and only != "a"
        if a_moves and i >= fa and not moved_a:
            ax += t["dir_a"][0] * t["jump_a"]; ay += t["dir_a"][1] * t["jump_a"]; moved_a = True
        if b_moves and i >= fb and not moved_b:
            bx += t["dir_b"][0] * t["jump_b"]; by += t["dir_b"][1] * t["jump_b"]; moved_b = True
        _draw(img, t["shape_a"], COLORS[t["color_a"]], ax, ay, t["size_a"])
        _draw(img, t["shape_b"], COLORS[t["color_b"]], bx, by, t["size_b"])
        vw.write(img)
    vw.release()
    # The question names one shape first; the label is about that shape, so the order cue cannot be
    # read off position or direction alone.
    named_first_moved_first = a_first if t["name_first"] == "a" else (not a_first)
    return dict(t, path=str(path), a_first=a_first, n_frames=n_frames, gap_frames=gap,
                f_first=f_first, f_second=f_second, fps=FPS, gt=int(named_first_moved_first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="videos per interval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--twins", default=None, help="render a frozen twin for a stratified subset")
    ap.add_argument("--n-twins", type=int, default=240, help="total twins, spread evenly over intervals")
    ap.add_argument("--pairs", type=int, default=240, help="order-flipped matched pairs, for the leak test")
    ap.add_argument("--solo", type=int, default=240, help="trials that also get one-shape-moves clips")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    rows, k = [], 0
    per_interval_twins = max(1, args.n_twins // len(INTERVALS_MS))
    per_interval_pairs = max(1, args.pairs // len(INTERVALS_MS))
    per_interval_solo = max(1, args.solo // len(INTERVALS_MS))
    # Draw every trial first, then assign the question's naming order so the label is balanced against
    # the nuisances, and only then render: name_first changes the prompt, not the video, so it has to be
    # settled before the metadata is written but costs nothing to decide late.
    trials = [(ms, j, sample_trial(rng, ms)) for ms in INTERVALS_MS for j in range(args.n)]
    balance_labels([t for _, _, t in trials], rng)

    for ms, j, t in trials:
        r = render(out / f"int_{ms}ms" / f"v3_{ms}_{k:05d}{EXT}", t)
        # stratified twins: the first few trials of EVERY interval, not the first 240 overall
        if args.twins and j < per_interval_twins:
            tw = render(Path(args.twins) / Path(r["path"]).name, t, frozen=True)
            r["twin_path"] = tw["path"]
        if args.pairs and j < per_interval_pairs:
            fl = render(out / f"int_{ms}ms" / f"v3_{ms}_{k:05d}_flip{EXT}", t, flip=True)
            r["flip_path"] = fl["path"]
        if args.solo and j < per_interval_solo:
            for who in ("a", "b"):
                s = render(out / f"int_{ms}ms" / f"v3_{ms}_{k:05d}_solo{who}{EXT}", t, only=who)
                r[f"solo_{who}_path"] = s["path"]
        rows.append(r)
        k += 1
    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({kk for r in rows for kk in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_tw = sum("twin_path" in r for r in rows)
    n_fl = sum("flip_path" in r for r in rows)
    n_so = sum("solo_a_path" in r for r in rows)
    print(f"rendered {len(rows)} unique videos (+{n_tw} twins, +{n_fl} flipped pairs, "
          f"+{2*n_so} one-shape-moves clips) -> {out}")


if __name__ == "__main__":
    main()
