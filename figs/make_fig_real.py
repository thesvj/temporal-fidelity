"""Figure for the real-footage appendix: one trial, five frames. CPU only.

Usage (from release_v3/): uv run --no-project --with numpy --with opencv-python-headless --with matplotlib \
    python figs/make_fig_real.py --out ../paper_long/figs/fig_real_trial.pdf
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_real as g  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="clips.csv")
ap.add_argument("--src", default="data/real_clips")
ap.add_argument("--top", default="b006")       # squirrel
ap.add_argument("--bottom", default="b081")    # guitarist (hands only)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rows = {r["burst"]: r for r in csv.DictReader(open(a.clips))}
A, B = g.load_burst(Path(a.src), rows[a.top]), g.load_burst(Path(a.src), rows[a.bottom])
K, f1, gap = g.K, 30, 30                      # 1 s lead, 1 s interval; the bottom panel moves first
fa, fb = f1 + gap, f1
shots = [(0, "both still"), (fb + 7, "bottom mid-burst"), (fb + K + 6, "bottom done, top still"),
         (fa + 7, "top mid-burst"), (fa + K + 20, "both done")]
fig, axs = plt.subplots(1, len(shots), figsize=(7.0, 1.72))
for ax, (i, lab) in zip(axs, shots):
    pa, pb = min(max(i - fa + 1, 0), K), min(max(i - fb + 1, 0), K)
    ax.imshow(cv2.cvtColor(np.vstack([A[pa], B[pb]]), cv2.COLOR_BGR2RGB))
    ax.axhline(240, color="white", lw=.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{i / 30:.2f} s", fontsize=7, pad=2)
    ax.set_xlabel(lab, fontsize=6.5, labelpad=2)
    for s in ax.spines.values():
        s.set_linewidth(.4)
fig.subplots_adjust(left=.005, right=.995, top=.9, bottom=.1, wspace=.04)
fig.savefig(a.out, dpi=220)
print("wrote", a.out, "| top:", rows[a.top]["author"], rows[a.top]["licence"], "| bottom:", rows[a.bottom]["author"], rows[a.bottom]["licence"])
