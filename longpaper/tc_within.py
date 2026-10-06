"""TempCompass yes/no: within-video margin AUC (only pairs of items from the same video with opposite gold answers)
and direction-item BAcc, real vs static clips, with a video bootstrap of the difference. CPU only.
  uv run --no-project --with numpy python longpaper/tc_within.py REAL.csv STATIC.csv [dims]"""
import csv
import sys
from collections import defaultdict

import numpy as np


def load(f):
    return {(r["video_id"], r["question"]): r for r in csv.DictReader(open(f))}


def wv(rows, vids):
    num = den = 0.0
    for v in vids:
        pos = [float(r["margin"]) for r in rows[v] if r["gold"] == "1"]
        neg = [float(r["margin"]) for r in rows[v] if r["gold"] == "0"]
        for p in pos:
            for n in neg:
                num += (p > n) + .5 * (p == n); den += 1
    return num / den if den else np.nan


def bacc(rs):
    y = np.array([r["gold"] == "1" for r in rs]); p = np.array([r["pred"] == "1" for r in rs])
    return (p[y].mean() + 1 - p[~y].mean()) / 2


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2]); dims = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    keys = [k for k in A if k in B and (dims is None or A[k]["dim"] in dims)]
    ra, rb = defaultdict(list), defaultdict(list)
    for k in keys:
        ra[k[0]].append(A[k]); rb[k[0]].append(B[k])
    V = list(ra); rng = np.random.default_rng(20261006)
    a, b = wv(ra, V), wv(rb, V); d = []
    for _ in range(2000):
        s = [V[i] for i in rng.integers(0, len(V), len(V))]
        ca, cb = defaultdict(list), defaultdict(list)
        for j, v in enumerate(s):
            ca[(v, j)] = ra[v]; cb[(v, j)] = rb[v]
        d.append(wv(ca, list(ca)) - wv(cb, list(cb)))
    dirA = [A[k] for k in keys if A[k]["dim"] == "direction"]; dirB = [B[k] for k in keys if B[k]["dim"] == "direction"]
    print(f"dims={dims} within real {a:.3f} static {b:.3f} diff {a-b:+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]"
          f" | direction BAcc real {bacc(dirA):.3f} static {bacc(dirB):.3f}")


if __name__ == "__main__":
    main()
