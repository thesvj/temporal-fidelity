"""Find candidate bursts in the downloaded source clips for the real-clip run. CPU only.

A burst is K+1 = 16 consecutive frames of one source video, shown as: frame 0 held still, frames 1..K
played, frame K held still. For that to be a clean event the camera must be static and uncut over the
burst, the change must be visible from the very first played frame, the pre and post stills must differ
clearly, and the change must be local (a person moving, not the whole image).

For every clip this scores every candidate start frame and keeps the best non-overlapping ones, with a
2:1 crop around the moving region (the panel is 480x240). It writes <src>/bursts.csv and contact-sheet
pages <src>/sheets/page_XX.jpg (frames 0, 1, 5, 10, 15 of each burst, cropped as it would be rendered)
for the human pass that picks bursts and names them. Nothing is selected here.

Usage: uv run --no-project --with numpy --with opencv-python-headless python screen_clips.py [--src data/real_src]
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

K = 15              # played frames per burst (0.5 s at 30 fps)
AH = 160            # analysis height
BG_MAX = 2.0        # mean abs grey-level change outside the moving region, frame 0 vs K: static camera
SHIFT_MAX = 0.35    # global shift (px at analysis size) between the two stills: a pan or zoom is not static
MAIN_MIN = 0.6      # share of the changed pixels in the largest connected blob: one mover, not a crowd
AREA = (.006, .15)  # fraction of the image that changes between the two stills: local, but not tiny
ONSET_MIN = .0015   # fraction of the image already changed one frame after onset
PER_CLIP = 2
SHOW = [0, 1, 5, 10, 15]
ROWS = 7


def load_gray(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (int(f.shape[1] * AH / f.shape[0]), AH)),
                                                 cv2.COLOR_BGR2GRAY), (5, 5), 0))
    cap.release()
    return np.array(out, np.uint8)


def score(g, s):
    a, b = g[s].astype(np.int16), g[s + K].astype(np.int16)
    mask = np.abs(b - a) > 22
    # any frame of the burst may differ from frame 0; the crop has to contain all of it
    union = mask.copy()
    for j in (4, 8, 12):
        union |= np.abs(g[s + j].astype(np.int16) - a) > 22
    area = mask.mean()
    if not AREA[0] <= area <= AREA[1]:
        return None
    grown = cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    (dx, dy), _ = cv2.phaseCorrelate(g[s].astype(np.float32), g[s + K].astype(np.float32))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(grown.astype(np.uint8))
    main = stats[1:, cv2.CC_STAT_AREA].max() / max(grown.sum(), 1) if n_lab > 1 else 0
    if np.hypot(dx, dy) > SHIFT_MAX or main < MAIN_MIN:
        return None
    big = lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bg = float(np.abs(b - a)[~grown].mean())
    onset = float((np.abs(g[s + 1].astype(np.int16) - a) > 14).mean())
    # the stills must really be still: the frames just before and just after belong to other motion,
    # which is fine, but a cut inside the burst is not
    step = np.abs(np.diff(g[s:s + K + 1].astype(np.int16), axis=0)).mean(axis=(1, 2))
    if bg > BG_MAX or onset < ONSET_MIN or step.max() > 6 * (np.median(step) + .3):
        return None
    ys, xs = np.where(big)
    return dict(start=s, area=round(float(area), 4), bg=round(bg, 2), onset=round(onset, 4),
                strength=round(float(np.abs(b - a)[mask].mean() * area), 3),
                box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))


def crop_for(box, w, h):
    """2:1 crop (in analysis pixels) containing the moving region with margin, as tight as allowed."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ch = max((y1 - y0) * 1.4, (x1 - x0) * 1.4 / 2, h * .4)
    ch = min(ch, h, w / 2)
    cw = 2 * ch
    left = min(max(cx - cw / 2, 0), w - cw)
    top = min(max(cy - ch / 2, 0), h - ch)
    return left / w, top / h, cw / w, ch / h      # fractions of the frame


def read_frames(path, idx):
    cap = cv2.VideoCapture(str(path))
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        out.append(cap.read()[1])
    cap.release()
    return out


def panel(frame, crop):
    H, W = frame.shape[:2]
    l, t, w, h = crop
    x, y = int(round(l * W)), int(round(t * H))
    return cv2.resize(frame[y:y + int(round(h * H)), x:x + int(round(w * W))], (480, 240), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/real_src")
    args = ap.parse_args()
    src = Path(args.src)
    meta = {re.sub(r"[^A-Za-z0-9]+", "_", r["title"][5:].rsplit(".", 1)[0])[:80]: r
            for r in csv.DictReader(open(src / "candidates.csv"))}
    rows = []
    clips = sorted((src / "clips").glob("*.mp4"))
    for n, p in enumerate(clips):
        g = load_gray(p)
        if len(g) < K + 40:
            continue
        cands = [c for s in range(10, len(g) - K - 10, 2) if (c := score(g, s))]
        cands.sort(key=lambda c: -c["strength"])
        kept = []
        for c in cands:
            if all(abs(c["start"] - k["start"]) >= 60 for k in kept):
                kept.append(c)
            if len(kept) == PER_CLIP:
                break
        m = meta.get(p.stem, {})
        for c in kept:
            crop = crop_for(c.pop("box"), g.shape[2], g.shape[1])
            rows.append(dict(c, file=p.name, category=m.get("category", ""),
                             crop=" ".join(f"{v:.4f}" for v in crop), licence=m.get("licence", ""),
                             author=m.get("author", ""), title=m.get("title", ""), page=m.get("page", "")))
        print(f"[{n + 1}/{len(clips)}] {p.name[:60]:60} {len(cands):4} candidates, kept {len(kept)}", flush=True)
    rows.sort(key=lambda r: (r["category"], r["file"], r["start"]))
    for i, r in enumerate(rows):
        r["idx"] = i
    with open(src / "bursts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx"] + [k for k in rows[0] if k != "idx"])
        w.writeheader(); w.writerows(rows)
    (src / "sheets").mkdir(exist_ok=True)
    for old in (src / "sheets").glob("page_*.jpg"):
        old.unlink()
    for pg in range(0, len(rows), ROWS):
        strips = []
        for r in rows[pg:pg + ROWS]:
            crop = [float(v) for v in r["crop"].split()]
            fr = [cv2.resize(panel(f, crop), (300, 150)) for f in
                  read_frames(src / "clips" / r["file"], [r["start"] + j for j in SHOW])]
            bar = np.full((16, 1500, 3), 30, np.uint8)
            cv2.putText(bar, f"#{r['idx']} {r['category']} | {r['file'][:60]} | f{r['start']} area {r['area']} "
                             f"bg {r['bg']} onset {r['onset']}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            strips += [bar, np.hstack(fr)]
        cv2.imwrite(str(src / "sheets" / f"page_{pg // ROWS:02d}.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(rows)} candidate bursts from {len({r['file'] for r in rows})} clips, "
          f"{(len(rows) + ROWS - 1) // ROWS} sheet pages -> {src / 'sheets'}")


if __name__ == "__main__":
    main()
