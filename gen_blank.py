"""
Blank-video prior-probe stimuli — Exp E for the rebuttal (P2uF W2).

Two sub-conditions x n videos, matching E1's look (480x480, gray-220, 30 fps,
~3.5 s):
  blank  — uniform gray, no shapes at all (question references absent objects)
  static — both shapes present at E1 positions but NEITHER ever moves

The E1 order question is asked unchanged. There is no correct answer grounded
in motion; a perceptual model should hedge or refuse, a prior-driven model
emits its default. Analysis: P(YES) / response distribution only ("a_first"
is a dummy so probe.py runs unmodified — ignore its "correct" column).

Usage: python gen_blank.py --out data/blank_videos --n 50
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

W, H, SHAPE_SIZE = 480, 480, 40
COLORS = {
    "red":   (0,   0,   255),
    "blue":  (255, 0,   0),
    "green": (0,   200, 0),
}
SHAPE_PAIRS = [
    ("red", "square", "blue",  "circle"),
    ("blue", "square", "green", "circle"),
    ("green", "circle", "red",  "square"),
]
FPS = 30
DURATION_S = 3.5


def _draw(frame: np.ndarray, shape: str, bgr: tuple, cx: int, cy: int):
    s = SHAPE_SIZE // 2
    if shape == "square":
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), bgr, -1)
    else:
        cv2.circle(frame, (cx, cy), s, bgr, -1)


def render(path: Path, condition: str,
           color_a: str, shape_a: str, color_b: str, shape_b: str):
    n_frames = int(DURATION_S * FPS)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(n_frames):
        frame = np.full((H, W, 3), 220, np.uint8)
        if condition == "static":
            _draw(frame, shape_a, COLORS[color_a], W // 3, H // 3)
            _draw(frame, shape_b, COLORS[color_b], 2 * W // 3, 2 * H // 3)
        writer.write(frame)
    writer.release()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/blank_videos")
    p.add_argument("--n",   type=int, default=50, help="videos per condition")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    rows = []

    for condition in ("blank", "static"):
        for i in range(args.n):
            ca, sa, cb, sb = random.choice(SHAPE_PAIRS)
            tag = f"{condition}_{i:04d}"
            path = out_dir / condition / f"{tag}.mp4"
            render(path, condition, ca, sa, cb, sb)
            rows.append({
                "path":         str(path),
                "condition":    condition,
                "interval_ms":  0,
                "fps":          FPS,
                "color_a":      ca, "shape_a": sa,
                "color_b":      cb, "shape_b": sb,
                "a_first":      False,   # dummy: lets probe.py run unmodified
                "simultaneous": False,
            })

    meta = out_dir / "metadata.csv"
    with open(meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Blank/static dataset ready: {len(rows)} videos -> {meta}")


if __name__ == "__main__":
    main()
