"""LLaVA-NeXT-Video — llava-hf/LLaVA-NeXT-Video-7B-DPO-hf (native HF)."""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, load_frames


class LLaVANextVideo(VideoModel):
    name = "llava-next-video"
    n_llm_layers = 32
    n_enc_layers = 24

    def __init__(self, n_frames: int = 8):
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._proc  = LlavaNextVideoProcessor.from_pretrained(hf)
        self._model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            hf, torch_dtype=torch.float16, device_map="auto",
        )

    def _inputs(self, video_path: Path, prompt: str):
        frames = load_frames(video_path, self.n)
        conv   = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text   = self._proc.apply_chat_template(conv, add_generation_prompt=True)
        return self._proc(text=text, videos=frames, return_tensors="pt").to(self._model.device)

    def ask(self, video_path: Path, prompt: str) -> str:
        inputs = self._inputs(video_path, prompt)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=64)
        return self._proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        inputs = self._inputs(video_path, "Describe the video.")
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        frames = load_frames(video_path, self.n)
        pixel_values = self._proc(text="", videos=frames, return_tensors="pt").pixel_values_videos
        pixel_values = pixel_values.to(self._model.device, torch.float16)
        with torch.no_grad():
            vision_out = self._model.model.vision_tower(
                pixel_values.flatten(0, 1), output_hidden_states=True,
            )
        return {l: vision_out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}
