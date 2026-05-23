"""Qwen2.5-VL — Qwen/Qwen2.5-VL-7B-Instruct (native HF, mRoPE temporal encoding)."""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, load_frames, _block_hooks, _remove_hooks


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
            hf, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self._model.eval()

    def _inputs(self, video_path: Path, prompt: str):
        from PIL import Image
        frames     = load_frames(video_path, self.n)
        pil_frames = [Image.fromarray(f) for f in frames]
        messages   = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._proc(text=[text], videos=[pil_frames], return_tensors="pt").to(self._model.device)

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
