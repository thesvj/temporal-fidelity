"""Build static (middle frame repeated) and blank (uniform gray) versions of TempCompass clips.

Each control clip keeps the source fps, frame count and resolution, so frame sampling is identical.
Usage: python make_tc_controls.py  (reads data/tempcompass/yes_no_order_direction.csv)
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRC = Path("data/tempcompass/videos")
OUT = {"static": Path("data/tempcompass/videos_static"), "blank": Path("data/tempcompass/videos_blank")}


def main():
    ids = pd.read_csv("data/tempcompass/yes_no_order_direction.csv", dtype={"video_id": str})["video_id"].unique()
    for d in OUT.values():
        d.mkdir(parents=True, exist_ok=True)
    missing = 0
    for vid in ids:
        src = SRC / f"{vid}.mp4"
        if not src.exists():
            missing += 1
            continue
        if all((OUT[k] / f"{vid}.mp4").exists() for k in OUT):
            continue
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            missing += 1
            continue
        h, w = frames[0].shape[:2]
        mid = frames[len(frames) // 2]
        gray = np.full_like(mid, 128)
        for kind, img in (("static", mid), ("blank", gray)):
            vw = cv2.VideoWriter(str(OUT[kind] / f"{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for _ in frames:
                vw.write(img)
            vw.release()
    print(f"controls built for {len(ids) - missing} clips; missing/unreadable source: {missing}")


if __name__ == "__main__":
    main()
