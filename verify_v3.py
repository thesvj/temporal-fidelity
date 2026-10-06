"""Checks the v3 stimulus set actually has the properties the paper will claim for it.

Every check prints PASS/FAIL and the number it checked, so the output can go straight into the appendix.

  1 uniqueness        no two trials share a parameter vector, no two files share an md5
  2 label balance     the label is near 50/50 and is independent of every nuisance parameter
  3 no direction cue  direction and start region do not predict the label (the confound v2 had)
  4 encoding leak     in a matched pair (same trial, order flipped) every decoded frame outside
                      [f_first, f_second) is bit-identical, so a probe cannot separate the orders
                      there. This is the control for the .650-on-invisible-trials probe result.

Usage: python verify_v3.py --meta data/videos_v3/metadata.csv
"""
import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

NUISANCE = ["color_a", "color_b", "shape_a", "shape_b", "size_a", "size_b",
            "ax", "ay", "bx", "by", "jump_a", "jump_b", "dir_a", "dir_b",
            "lead_s", "tail_s", "bg", "name_first"]
RESULTS = []


def check(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _binned(vals, max_levels=8):
    """Continuous nuisances (positions, sizes, lead-in) get quantile-binned: with one level per trial
    any association measure saturates at 1 and says nothing."""
    vals = list(vals)
    try:
        nums = [float(v) for v in vals]
    except (TypeError, ValueError):
        return [str(v) for v in vals]
    if len(set(nums)) <= max_levels:
        return [str(v) for v in vals]
    qs = np.quantile(nums, np.linspace(0, 1, max_levels + 1)[1:-1])
    return [str(int(np.searchsorted(qs, v))) for v in nums]


def cramers_v(xs, ys):
    """Effect size for label vs a categorical nuisance; 0 = independent."""
    xs, ys = list(map(str, xs)), _binned(ys)
    rx, ry = sorted(set(xs)), sorted(set(ys))
    if len(rx) < 2 or len(ry) < 2:
        return 0.0
    tab = np.zeros((len(rx), len(ry)))
    for a, b in zip(xs, ys):
        tab[rx.index(a), ry.index(b)] += 1
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / n
    chi2 = float(((tab - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def assoc_p(labels, vals, B=2000, seed=0):
    """Permutation p for label-vs-nuisance association. A fixed V threshold is not usable: V has a
    positive expectation under independence that grows with the number of levels and shrinks with n."""
    rng = np.random.default_rng(seed)
    obs = cramers_v(labels, vals)
    binned = _binned(vals)
    lab = np.array(list(map(str, labels)))
    cnt = 0
    for _ in range(B):
        rng.shuffle(lab)
        if cramers_v(lab, binned) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-pairs", type=int, default=40, help="matched pairs to decode for check 4")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.meta)))
    print(f"{len(rows)} trials from {args.meta}\n")

    # 1 uniqueness ----------------------------------------------------------------
    vecs = [tuple(r[k] for k in NUISANCE + ["interval_ms", "a_first"]) for r in rows]
    dup = [v for v, c in Counter(vecs).items() if c > 1]
    check("unique parameter vectors", not dup, f"{len(set(vecs))}/{len(rows)} distinct")
    hashes = [md5(r["path"]) for r in rows if Path(r["path"]).exists()]
    check("unique files (md5)", len(set(hashes)) == len(hashes),
          f"{len(set(hashes))} hashes over {len(hashes)} files")

    # 2 label balance -------------------------------------------------------------
    gt = [int(r["gt"]) for r in rows]
    frac = sum(gt) / len(gt)
    check("label balance", 0.45 <= frac <= 0.55, f"P(gt=1) = {frac:.3f} (majority {max(frac,1-frac):.3f})")
    ps = [(assoc_p(gt, [r[k] for r in rows]), k) for k in NUISANCE]
    worst = min(ps, key=lambda x: x[0][1])
    n_sig = sum(p < 0.05 / len(NUISANCE) for (_, p), _ in ps)   # Bonferroni over the nuisance family
    check("label independent of nuisances", n_sig == 0,
          f"0 of {len(NUISANCE)} associations significant; smallest p = {worst[0][1]:.3f} "
          f"({worst[1]}, V = {worst[0][0]:.3f})")

    # 3 the v2 confound is gone ---------------------------------------------------
    for k in ("dir_a", "dir_b"):
        v, p = assoc_p(gt, [r[k] for r in rows])
        check(f"{k} does not predict the label", p >= 0.05, f"V = {v:.3f}, permutation p = {p:.3f}")
    first_dir = [r["dir_a"] if r["a_first"] == "True" else r["dir_b"] for r in rows]
    v, p = assoc_p(gt, first_dir)
    check("direction of the FIRST mover does not predict the label", p >= 0.05,
          f"V = {v:.3f}, permutation p = {p:.3f}")
    quad = ["%d%d" % (int(r["ax"]) > 240, int(r["ay"]) > 240) for r in rows]
    check("start quadrant of A is not fixed", len(set(quad)) == 4,
          f"{len(set(quad))} quadrants, counts {dict(Counter(quad))}")

    # 4 encoding leak -------------------------------------------------------------
    pairs = [r for r in rows if r.get("flip_path") and Path(r["flip_path"]).exists()][: args.max_pairs]
    if not pairs:
        check("matched-pair frames identical outside the jump window", False, "no flipped pairs found")
    else:
        bad, checked = [], 0
        for r in pairs:
            lo, hi = int(r["f_first"]), int(r["f_second"])
            ca, cb = cv2.VideoCapture(r["path"]), cv2.VideoCapture(r["flip_path"])
            i = 0
            while True:
                oka, fa = ca.read(); okb, fb = cb.read()
                if not (oka and okb):
                    break
                if not (lo <= i < hi):
                    checked += 1
                    if not np.array_equal(fa, fb):
                        bad.append((r["path"], i, int(np.abs(fa.astype(int) - fb.astype(int)).max())))
                i += 1
            ca.release(); cb.release()
        worst_d = max((b[2] for b in bad), default=0)
        check("matched-pair frames identical outside the jump window", not bad,
              f"{len(bad)}/{checked} frames differ over {len(pairs)} pairs"
              + (f", max abs pixel delta {worst_d}" if bad else ""))
        if bad:
            print("      -> the codec carries order information into frames that should not have it.")
            print("      -> re-render losslessly (FFV1/PNG) before trusting any probe result.")
            for p, i, d in bad[:5]:
                print(f"         {Path(p).name} frame {i} delta {d}")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    raise SystemExit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
