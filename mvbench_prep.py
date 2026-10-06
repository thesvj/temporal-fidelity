"""Harness validation on MVBench: build the item list and the clips the adapters read. CPU only.

Task -> folder, input type and bound flag follow the official MVBench evaluation (VideoChat2 `mvbench.ipynb`).
Fine-grained Pose (NTU RGB+D) is not distributed with the benchmark, so 19 of the 20 tasks are built.
Nothing is cut or re-encoded: bounds (start/end seconds) and TVQA frame folders are passed to each model's own
loader, as the official evaluation scripts do.

Usage: python mvbench_prep.py --root data/mvbench --out data/mvbench/items.csv
"""
import argparse
import csv
import json
from pathlib import Path

TASKS = {  # json, folder (relative to video/), type, bound
    "Action Sequence": ("action_sequence.json", "star/Charades_v1_480", "video", True),
    "Action Prediction": ("action_prediction.json", "star/Charades_v1_480", "video", True),
    "Action Antonym": ("action_antonym.json", "ssv2_video", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos", "video", False),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test", "video", False),
    "Object Existence": ("object_existence.json", "clevrer/video_validation", "video", False),
    "Object Interaction": ("object_interaction.json", "star/Charades_v1_480", "video", True),
    "Object Shuffle": ("object_shuffle.json", "perception/videos", "video", False),
    "Moving Direction": ("moving_direction.json", "clevrer/video_validation", "video", False),
    "Action Localization": ("action_localization.json", "sta/sta_video", "video", True),
    "Scene Transition": ("scene_transition.json", "scene_qa/video", "video", False),
    "Action Count": ("action_count.json", "perception/videos", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation", "video", False),
    "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation", "video", False),
    "State Change": ("state_change.json", "perception/videos", "video", False),
    "Character Order": ("character_order.json", "perception/videos", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation", "video", False),
}


def find(video_root, folder, name):
    """The official folder if it exists, else the unique file of that name anywhere under video/."""
    for p in (video_root / folder / name, video_root / "data0613" / folder / name):  # data0613: files added later
        if p.exists():
            return p
    hits = [q for q in video_root.rglob(Path(name).name) if folder.split("/")[0] in str(q)]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvbench")
    ap.add_argument("--out", default="data/mvbench/items.csv")
    a = ap.parse_args()
    root = Path(a.root); vroot = root / "video"
    rows, missing = [], []
    for task, (js, folder, typ, bound) in TASKS.items():
        items = json.load(open(root / "json" / js))
        for k, it in enumerate(items):
            cands = it["candidates"]
            src = (vroot / folder / it["video"]) if typ == "frame" else find(vroot, folder, it["video"])
            if src is None or not src.exists():
                missing.append((task, it["video"])); continue
            rows.append(dict(task=task, idx=k, video=str(src), type=typ,
                             start=it["start"] if bound else "", end=it["end"] if bound else "",
                             question=it["question"], candidates=json.dumps(cands),
                             answer="ABCDE"[cands.index(it["answer"])]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows:
        by[r["task"]] = by.get(r["task"], 0) + 1
    print(json.dumps(by, indent=1))
    print("ITEMS", len(rows), "MISSING", len(missing), missing[:10])


if __name__ == "__main__":
    main()
