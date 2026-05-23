"""VideoLLaMA3 — DAMO-NLP-SG/VideoLLaMA3-7B (SigLIP + DiffFP + Qwen2.5, trust_remote_code)."""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, _block_hooks, _remove_hooks


class VideoLLaMA3Adapter(VideoModel):
    name = "videollama3"
    n_llm_layers = 28   # Qwen2.5-7B
    n_enc_layers = 27   # SigLIP ViT-400M

    def __init__(self, n_frames: int = 8):
        from transformers import AutoModelForCausalLM, AutoProcessor
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._proc  = AutoProcessor.from_pretrained(hf, trust_remote_code=True)
        self._patch_processor_kwargs()
        self._model = AutoModelForCausalLM.from_pretrained(
            hf, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", attn_implementation="sdpa",
        )
        self._model.eval()

    def _patch_processor_kwargs(self):
        """Add common_kwargs annotation removed in transformers 5.x."""
        import sys
        from typing import Optional, TypedDict, Union
        from transformers.processing_utils import ProcessingKwargs
        proc_mod = sys.modules.get(type(self._proc).__module__)
        if proc_mod is None:
            return
        cls = getattr(proc_mod, "Videollama3Qwen2ProcessorKwargs", None)
        if cls is None or not isinstance(cls, type):
            return
        for k, v in ProcessingKwargs.__annotations__.items():
            cls.__annotations__.setdefault(k, v)
        if "common_kwargs" not in cls.__annotations__:
            class _CommonKwargs(TypedDict, total=False):
                return_tensors: Optional[Union[str]]
            cls.__annotations__["common_kwargs"] = _CommonKwargs

    def _make_inputs(self, video_path: Path, prompt: str):
        conversation = [{"role": "user", "content": [
            {"type": "video", "video": {"video_path": str(video_path), "max_frames": self.n}},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._proc(conversation=conversation, return_tensors="pt").to(self._model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
        return inputs

    def ask(self, video_path: Path, prompt: str) -> str:
        inputs = self._make_inputs(video_path, prompt)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=64)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._proc.decode(gen, skip_special_tokens=True).strip()

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._make_inputs(video_path, "Describe the video.")
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._make_inputs(video_path, "Describe the video.")
        blocks = list(self._model.model.vision_encoder.encoder.layers)
        captured, handles = _block_hooks(blocks, layers)
        with torch.no_grad():
            self._model(**inputs)
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=0, keepdim=True) for l in layers}
