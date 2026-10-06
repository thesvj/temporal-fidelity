"""Log the frame indices each Qwen2.5-VL arm delivers (models/qwen25_vl.arm_frames), in delivery order. CPU only.

Usage: QWEN_ARM=shift python record_arm_frames.py --meta data/videos_v3/metadata.csv --out results_arm_shift_v3
Writes <out>/sampled_frames_qwen2.5-vl.jsonl (the name analyze_v3.load_recorded expects) and copies the metadata.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from models.qwen25_vl import arm_frames

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(a.meta)))
with open(out / "sampled_frames_qwen2.5-vl.jsonl", "w") as fh:
    for r in rows:
        idx, fps = arm_frames(r["path"], 8)
        fh.write(json.dumps({"model": "qwen2.5-vl", "arm": os.environ.get("QWEN_ARM", ""), "path": r["path"],
                             "n_frames": int(r["n_frames"]), "idx": [int(i) for i in idx], "fps": fps}) + "\n")
shutil.copy(a.meta, out / "videos_v3_metadata.csv")
print("RECORDED", os.environ.get("QWEN_ARM", "") or "harness", len(rows), "->", out)
