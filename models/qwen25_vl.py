"""Qwen2.5-VL — Qwen/Qwen2.5-VL-7B-Instruct (native HF, mRoPE temporal encoding)."""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, load_frames, _block_hooks, _remove_hooks


def _mem_kwargs():
    # single-GPU runs of the 72B/78B checkpoints: cap GPU memory, spill the rest to CPU (exact bf16)
    import os
    g = os.environ.get("MAX_GPU_GIB")
    if not g:
        return {}
    kw = {"max_memory": {0: f"{g}GiB", "cpu": "400GiB"}}
    if os.environ.get("QUANT") == "fp8":
        from transformers import FineGrainedFP8Config
        kw = {"quantization_config": FineGrainedFP8Config()}
    return kw


def _read(path, idx):
    import decord
    decord.bridge.set_bridge("native")
    return decord.VideoReader(str(path), ctx=decord.cpu(0)).get_batch(list(idx)).asnumpy()


def arm_frames(path, n=8):
    """Frame indices (in delivery order) and the fps handed to the processor, for the arm named by QWEN_ARM.

    Qwen2.5-VL fuses delivered frames in pairs (temporal patch size 2), so which two frames share a patch
    is part of what the model receives. The arms change one thing each:
      ""      harness, as in the paper: n uniform frames, processor default fps (2.0)
      "ts"    the same frames, fps = n / duration, so only the temporal position ids change
      "shift" [i0, i0, i1, ..., i(n-2)]: every frame one slot later, so every patch boundary moves by one frame
      "shift2" [i0, i0, i0, i1, ..., i(n-3)]: two slots later; same duplication and loss of late frames, pairing unchanged
      "native" 2 fps sampling as in qwen_vl_utils (frame count, alignment and timestamps all change)
    """
    import os
    import decord
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    total, vfps = len(vr), vr.get_avg_fps()
    arm = os.environ.get("QWEN_ARM", "")
    uni = [int(i * total / n) for i in range(n)]
    if arm == "":
        return uni, None
    if arm == "ts":
        return uni, n / (total / vfps)
    if arm == "shift":
        return [uni[0]] + uni[:-1], None
    if arm == "shift2":                                  # control: two slots later, so the pairing is unchanged
        return [uni[0], uni[0]] + uni[:-2], None
    if arm == "native":
        k = int(min(max(total / vfps * 2.0, 4), (min(768, total) // 2) * 2, total) // 2) * 2
        idx = torch.linspace(0, total - 1, k).round().long().tolist()
        return idx, k / total * vfps
    raise ValueError(f"unknown QWEN_ARM {arm!r}")


class Qwen25VL(VideoModel):
    name = "qwen2.5-vl"
    n_llm_layers = 28   # Qwen2.5-7B
    n_enc_layers = 32   # ViT blocks

    def __init__(self, n_frames: int = 8):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._proc  = AutoProcessor.from_pretrained(hf)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf, torch_dtype=torch.bfloat16, device_map="auto", **_mem_kwargs(),
        )
        self._model.eval()

    def _inputs(self, video_path: Path, prompt: str):
        from PIL import Image
        idx, fps   = arm_frames(video_path, self.n)
        frames     = _read(video_path, idx)
        pil_frames = [Image.fromarray(f) for f in frames]
        messages   = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kw = {} if fps is None else {"fps": [fps]}
        return self._proc(text=[text], videos=[pil_frames], return_tensors="pt", **kw).to(self._model.device)

    def ask(self, video_path: Path, prompt: str) -> str:
        inputs = self._inputs(video_path, prompt)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=64)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._inputs(video_path, "Describe the video.")
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._inputs(video_path, "Describe the video.")
        pv  = inputs.get("pixel_values_videos", inputs.get("pixel_values"))
        thw = inputs.get("video_grid_thw",      inputs.get("image_grid_thw"))
        captured, handles = _block_hooks(list(self._model.visual.blocks), layers)
        with torch.no_grad():
            self._model.visual(pv, grid_thw=thw)
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=0, keepdim=True) for l in layers}


class Qwen25VL72B(Qwen25VL):
    """Qwen2.5-VL-72B-Instruct — scaling control (same arch as 7B, 72B params)."""
    name = "qwen2.5-vl-72b"
    n_llm_layers = 80   # Qwen2.5-72B
    n_enc_layers = 32   # ViT blocks


class Qwen3VL8B(Qwen25VL):
    """Qwen3-VL-8B-Instruct — second model family with temporal patch size 2, for the frame-pairing arms only."""
    name = "qwen3-vl-8b"

    def __init__(self, n_frames: int = 8):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._proc  = AutoProcessor.from_pretrained(hf)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(hf, torch_dtype=torch.bfloat16, device_map="auto")
        self._model.eval()

    def _inputs(self, video_path: Path, prompt: str):
        # Qwen3-VL's video processor resamples a bare frame list (8 frames became 4: positions 0, 2, 5, 7) and
        # needs the frames' times for its timestamp text. Deliver exactly the arm's frames, with their true times.
        import decord
        from PIL import Image
        idx, _ = arm_frames(video_path, self.n)
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        total, vfps = len(vr), vr.get_avg_fps()
        pil_frames = [Image.fromarray(f) for f in _read(video_path, idx)]
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        meta = [dict(total_num_frames=total, fps=vfps, frames_indices=list(idx), duration=total / vfps)]
        out = self._proc(text=[text], videos=[pil_frames], return_tensors="pt", do_sample_frames=False, video_metadata=meta)
        assert int(out["video_grid_thw"][0][0]) * 2 == len(idx), f"delivered {out['video_grid_thw']} for {len(idx)} frames"
        return out.to(self._model.device)
