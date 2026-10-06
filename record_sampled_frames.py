"""Record the frame indices each adapter actually decodes, instead of re-implementing its sampler.

Visibility ("does any frame the model sees fall between the two jumps?") is the per-trial variable the
paper conditions on, so it must come from the loader the model really runs, not from our reading of it.
This patches decord.VideoReader.get_batch, calls each model's own video-loading function on CPU (no
weights are loaded), and writes one JSON line per clip: {"model", "path", "n_frames", "idx"}.

Run once per venv on a login node:
  .venv-video-llama2/bin/python record_sampled_frames.py --model video-llama2
  .venv-videochat-flash/bin/python record_sampled_frames.py --model videochat-flash
  .venv-molmo2/bin/python record_sampled_frames.py --model molmo2
  .venv-qwen2.5-vl/bin/python record_sampled_frames.py --model qwen2.5-vl   # base.load_frames(8)
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import decord  # noqa: E402

CALLS = []
_orig = decord.VideoReader.get_batch


def _rec(self, indices):
    CALLS.append([int(i) for i in (indices.tolist() if hasattr(indices, "tolist") else indices)])
    return _orig(self, indices)


decord.VideoReader.get_batch = _rec


def loader(model, n):
    if model == "video-llama2":
        from models.video_llama2 import _ensure_videollama2
        _ensure_videollama2()
        from transformers import AutoConfig, CLIPImageProcessor
        from videollama2.mm_utils import process_video
        from models.base import HF_IDS
        cfg = AutoConfig.from_pretrained(HF_IDS[model])
        nf = getattr(cfg, "num_frames", None) or n   # model_init passes model.config.num_frames
        proc = CLIPImageProcessor.from_pretrained(cfg.mm_vision_tower)
        print(f"video-llama2 num_frames={nf}", flush=True)
        return lambda p: process_video(p, processor=proc, aspect_ratio=None, num_frames=nf)
    if model == "videochat-flash":
        import importlib
        import transformers.dynamic_module_utils as dmu
        from transformers import AutoConfig
        from models.base import HF_IDS
        g = dmu.get_imports
        dmu.get_imports = lambda f: [i for i in g(f) if i != "flash_attn"]
        AutoConfig.from_pretrained(HF_IDS[model], trust_remote_code=True)
        mod = next(m for k, m in sys.modules.items() if k.endswith(".mm_utils") and hasattr(m, "load_video"))

        # load_video routes .avi through PyAV (read_frames_av), not decord, so get_batch never fires.
        # Same arguments load_video passes (min_num_frames=64 overrides the wrapper's max_num_frames=8);
        # the returned indices are the frames the model receives. PyAV and decord agree pixel-exactly
        # on these FFV1 clips (checked: max abs diff 0).
        def vcf(p):
            _, idx, _, _ = mod.read_frames_av(video_path=p, num_frames=n, sample="dynamic_fps1",
                                              fix_start=None, min_num_frames=64, max_num_frames=n,
                                              local_num_frames=8)
            CALLS.append([int(i) for i in idx])
        return vcf
    if model == "molmo2":
        from transformers import AutoProcessor
        from models.base import HF_IDS
        from models.molmo2 import Molmo2
        proc = AutoProcessor.from_pretrained(HF_IDS[model], trust_remote_code=True)
        m = Molmo2.__new__(Molmo2)
        m._proc, m.n = proc, n
        m._model = type("D", (), {"device": "cpu"})()
        return lambda p: m._inputs(Path(p), "Describe the video.")
    from models.base import load_frames
    return lambda p: load_frames(Path(p), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", default="data/videos_v3/metadata.csv")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="results_v3/sampled_frames.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.meta)))[: a.limit]
    load = loader(a.model, a.n)
    with open(a.out, "a") as fh:
        for i, r in enumerate(rows):
            CALLS.clear()
            load(r["path"])
            if not CALLS:
                raise RuntimeError(f"{a.model}: loader bypassed decord.get_batch — cannot record")
            idx = sorted({j for c in CALLS for j in c})
            fh.write(json.dumps({"model": a.model, "n": a.n, "path": r["path"],
                                 "n_frames": int(r["n_frames"]), "idx": idx}) + "\n")
            if i < 2:
                print(a.model, r["path"], r["n_frames"], idx, flush=True)
    print(f"RECORDED {a.model} {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
