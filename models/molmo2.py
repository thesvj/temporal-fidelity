"""Molmo 2 — allenai/Molmo2-8B (SigLIP2 + Qwen3-8B, trust_remote_code).

Requires transformers~=4.57 — the remote modeling code is incompatible with 5.x.
The run script pins the correct version in the .venv-molmo2 venv after uv sync.
"""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, load_frames, _block_hooks, _remove_hooks


def _ensure_molmo2_files():
    """Download the modeling file if missing from HF cache."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("allenai/Molmo2-8B", "modeling_molmo2.py")
    except Exception:
        pass


class Molmo2(VideoModel):
    name = "molmo2"
    n_llm_layers = 36   # Qwen3-8B
    n_enc_layers = 25   # SigLIP2 ViT-400M

    def __init__(self, n_frames: int = 8):
        _ensure_molmo2_files()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._proc  = AutoProcessor.from_pretrained(
            hf, trust_remote_code=True, dtype="auto", device_map="auto",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            hf, trust_remote_code=True, dtype="auto", device_map="auto",
        )
        self._model.eval()

    def _inputs(self, video_path: Path, prompt: str) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._proc.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        return {k: v.to(self._model.device) for k, v in inputs.items()}

    def ask(self, video_path: Path, prompt: str) -> str:
        inputs = self._inputs(video_path, prompt)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=64)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._inputs(video_path, "Describe the video.")
        with torch.inference_mode():
            out = self._model(**inputs, output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._inputs(video_path, "Describe the video.")
        vb = self._model.model.vision_backbone
        blocks = list(vb.image_vit.transformer.resblocks)
        captured, handles = _block_hooks(blocks, layers)
        with torch.inference_mode():
            self._model(**inputs)
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=0, keepdim=True) for l in layers}
