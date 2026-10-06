"""Log the frame indices delivered under each model's reference preprocessing. CPU only.

  --kind qwen    Qwen2.5-VL reference sampling (2 fps, as models/qwen25_vl.arm_frames with QWEN_ARM=native)
  --kind ivl32   InternVL2.5 model-card sampling (midpoints of 32 segments, models/internvl25.ref_index)
Writes <out>/sampled_frames_qwen2.5-vl.jsonl: analyze_v3 looks up every uniform-8 adapter under that name
(RECORDED_SOURCE), so each reference run gets its own results directory. Copies the metadata.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import decord

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["qwen", "ivl32"], required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(a.meta)))
if a.kind == "qwen":
    os.environ["QWEN_ARM"] = "native"
    from models.qwen25_vl import arm_frames
else:
    from models.internvl25 import ref_index
with open(out / "sampled_frames_qwen2.5-vl.jsonl", "w") as fh:
    for r in rows:
        if a.kind == "qwen":
            idx, _ = arm_frames(r["path"], 8)
        else:
            idx = ref_index(len(decord.VideoReader(r["path"])))
        fh.write(json.dumps({"model": "qwen2.5-vl", "ref": a.kind, "path": r["path"], "n_frames": int(r["n_frames"]),
                             "idx": [int(i) for i in idx]}) + "\n")
shutil.copy(a.meta, out / "videos_v3_metadata.csv")
print("RECORDED", a.kind, len(rows), "->", out)
