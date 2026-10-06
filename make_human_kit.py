"""Human reference on the frames a model receives: build the rating material. CPU only.

For 50 shape trials and 50 real-footage trials (visible under uniform 8-frame sampling, stratified over the
12 intervals, labels balanced, one member per order-swapped pair), write the eight frames the 8-frame loader
decoded as one strip, with the order question. Humans see exactly what LLaVA-NeXT-Video, Qwen2.5-VL and
InternVL2.5 are given, no more. Output: <out>/items.json (question, strip file; the key is in key.json, kept
apart from what the rater sees) and <out>/strips/*.jpg.

Usage: python make_human_kit.py --out human_kit [--n 50] [--seed 0]
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

import analyze_v3 as av

P_ORDER = "Did the {n1} move BEFORE the {n2}?"


def names(r):
    if "noun_a" in r:
        a, b = r["noun_a"], r["noun_b"]
    else:
        a, b = f"{r['color_a']} {r['shape_a']}", f"{r['color_b']} {r['shape_b']}"
    return (a, b) if r["name_first"] == "a" else (b, a)


def pick(rows, n, rng):
    """n visible trials: as even as possible over intervals, half gt=1, at most one member per pair."""
    by = {}
    for r in rows:
        by.setdefault(int(r["interval_ms"]), []).append(r)
    out, seen, want = [], set(), {"0": n // 2, "1": n - n // 2}
    ints = sorted(by)
    for v in by.values():
        rng.shuffle(v)
    i = 0
    while len(out) < n and any(by.values()):
        ms = ints[i % len(ints)]; i += 1
        while by[ms]:
            r = by[ms].pop()
            key = r.get("pair_id") or r["path"]
            if key in seen or want[r["gt"]] == 0:
                continue
            seen.add(key); want[r["gt"]] -= 1; out.append(r)
            break
    return out


def strip(path, idx, h=240):
    cap = cv2.VideoCapture(str(path))
    fr = []
    for k, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        f = cv2.resize(cap.read()[1], (h, h), interpolation=cv2.INTER_AREA)
        cv2.putText(f, str(k + 1), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 3)
        cv2.putText(f, str(k + 1), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
        fr.append(f)
    cap.release()
    gap = np.full((h, 6, 3), 255, np.uint8)
    return np.hstack([x for f in fr for x in (f, gap)][:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape-meta", default="../data/videos_v3/metadata.csv")
    ap.add_argument("--shape-res", default="../results_v3")
    ap.add_argument("--real-meta", default="data/videos_real/metadata.csv")
    ap.add_argument("--real-res", default="results_real")
    ap.add_argument("--root-shape", default="..", help="directory the shape paths are relative to")
    ap.add_argument("--out", default="human_kit")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); (out / "strips").mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for fam, meta, res, root in (("shape", a.shape_meta, a.shape_res, a.root_shape), ("real", a.real_meta, a.real_res, ".")):
        av.RECORDED.clear(); av.load_recorded(res)
        rows = [r for r in csv.DictReader(open(meta)) if av.visible(r, "qwen2.5-vl")]
        for j, r in enumerate(pick(rows, a.n, rng)):
            idx = av.sampled(r, "qwen2.5-vl")
            iid = f"{fam}_{j:02d}"
            cv2.imwrite(str(out / "strips" / f"{iid}.jpg"), strip(Path(root) / r["path"], idx), [cv2.IMWRITE_JPEG_QUALITY, 88])
            n1, n2 = names(r)
            items.append(dict(id=iid, family=fam, question=P_ORDER.format(n1=n1, n2=n2)))
            key[iid] = dict(gt=int(r["gt"]), path=r["path"], interval_ms=int(r["interval_ms"]), frames=idx)
    rng.shuffle(items)
    json.dump(items, open(out / "items.json", "w"), indent=1)
    json.dump(key, open(out / "key.json", "w"), indent=1)
    print(len(items), "items ->", out, "| gt=1:", sum(k["gt"] for k in key.values()))


if __name__ == "__main__":
    main()
