"""
Synthetic video stimulus generation for temporal probing experiments.

E001/E002/E003: --fps 30
E004 (frame-rate control): --fps 8 16 30
E003 (simultaneity): --simultaneity
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

# Frame-aligned at 30 fps (33.3 ms/frame), log-spaced.
# Every value produces a visually distinct stimulus at the reference rate.
# At lower fps (E004), small intervals collapse to same-frame — by design,
# revealing how frame rate limits temporal resolution.
INTERVALS_MS = [33, 67, 100, 200, 333, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

# Simultaneity offsets: 0 = truly simultaneous, rest span 1–10 frames at 30 fps.
SIM_OFFSETS_MS = [0, 33, 67, 100, 200, 333]

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


def _draw(frame: np.ndarray, shape: str, bgr: tuple, cx: int, cy: int):
    s = SHAPE_SIZE // 2
    if shape == "square":
        cv2.rectangle(frame, (cx - s, cy - s), (cx + s, cy + s), bgr, -1)
    else:
        cv2.circle(frame, (cx, cy), s, bgr, -1)


def render_video(
    path: Path,
    interval_ms: int,
    fps: int,
    color_a: str, shape_a: str,
    color_b: str, shape_b: str,
    a_first: bool,
    simultaneous: bool = False,
):
    """Render one stimulus video: shape A moves left, shape B moves right."""
    pad_s = 1.0                          # 1 s lead-in before first event
    duration_s = pad_s + max(interval_ms / 1000, 0.1) + 1.5
    n_frames = int(duration_s * fps)

    frame_a = int(pad_s * fps)
    frame_b = frame_a if simultaneous else frame_a + int(interval_ms / 1000 * fps)
    if not a_first and not simultaneous:
        frame_a, frame_b = frame_b, frame_a

    bgr_a, bgr_b = COLORS[color_a], COLORS[color_b]
    ax, ay = W // 3,     H // 3
    bx, by = 2 * W // 3, 2 * H // 3
    moved_a = moved_b = False

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i in range(n_frames):
        frame = np.full((H, W, 3), 220, np.uint8)
        if i >= frame_a and not moved_a:
            ax -= 60; moved_a = True
        if i >= frame_b and not moved_b:
            bx += 60; moved_b = True
        _draw(frame, shape_a, bgr_a, ax, ay)
        _draw(frame, shape_b, bgr_b, bx, by)
        writer.write(frame)

    writer.release()


def generate_dataset(
    out_dir: Path,
    n_per_condition: int = 250,
    fps_list: list[int] | None = None,
    simultaneity: bool = False,
) -> Path:
    fps_list = fps_list or [30]
    intervals = SIM_OFFSETS_MS if simultaneity else INTERVALS_MS
    rows = []

    total = len(fps_list) * len(intervals) * n_per_condition
    done = 0

    for fps in fps_list:
        for interval_ms in intervals:
            for i in range(n_per_condition):
                ca, sa, cb, sb = random.choice(SHAPE_PAIRS)
                a_first = random.choice([True, False])
                sim = simultaneity and interval_ms == 0

                tag = f"fps{fps}_int{interval_ms}_{'af' if a_first else 'bf'}_{i:04d}"
                path = out_dir / f"fps{fps}" / f"int_{interval_ms}ms" / f"{tag}.mp4"
                render_video(path, interval_ms, fps, ca, sa, cb, sb, a_first, sim)

                rows.append({
                    "path":         str(path),
                    "interval_ms":  interval_ms,
                    "fps":          fps,
                    "color_a":      ca, "shape_a": sa,
                    "color_b":      cb, "shape_b": sb,
                    "a_first":      a_first,
                    "simultaneous": sim,
                })

                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{total}", flush=True)

    meta = out_dir / "metadata.csv"
    with open(meta, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset ready: {len(rows)} videos → {meta}")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out",          default="data/videos")
    p.add_argument("--n",            type=int,   default=250)
    p.add_argument("--fps",          nargs="+",  type=int, default=[30])
    p.add_argument("--simultaneity", action="store_true",
                   help="Generate E003 stimuli (0/100/200/300 ms offsets)")
    args = p.parse_args()
    generate_dataset(Path(args.out), args.n, args.fps, args.simultaneity)
