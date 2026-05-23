"""VideoLLaMA2 — DAMO-NLP-SG/VideoLLaMA2-7B (auto-clones repo)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from models.base import VideoModel, HF_IDS, CACHE


def _ensure_videollama2() -> Path:
    repo = CACHE / "VideoLLaMA2"
    if not (repo / "videollama2").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print("Cloning VideoLLaMA2 repository ...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/DAMO-NLP-SG/VideoLLaMA2.git", str(repo)],
            check=True,
        )
        subprocess.run(
            ["uv", "pip", "install", "-e", str(repo),
             "--no-deps", "--quiet", "--python", sys.executable],
            check=True,
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


class VideoLLaMA2(VideoModel):
    name = "video-llama2"
    n_llm_layers = 32
    n_enc_layers = 24

    def __init__(self, n_frames: int = 8):
        _ensure_videollama2()
        from videollama2 import model_init
        self._model, self._proc, self._tok = model_init(HF_IDS[self.name])
        self.n = n_frames

    def ask(self, video_path: Path, prompt: str) -> str:
        from videollama2 import mm_infer
        video_tensor = self._proc["video"](str(video_path))
        return mm_infer(video_tensor, prompt, self._model, self._tok, modal="video")

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        import copy
        from videollama2.conversation import conv_templates
        video_tensor = self._proc["video"](str(video_path))
        video_tensor = video_tensor.to(self._model.device, torch.float16)
        conv = copy.deepcopy(conv_templates["mistral"])
        conv.messages = list(conv.messages)
        conv.append_message(conv.roles[0], "<video>\nDescribe the video.")
        conv.append_message(conv.roles[1], None)
        input_ids = (
            self._tok(conv.get_prompt(), return_tensors="pt")
            .input_ids.to(self._model.device)
        )
        with torch.no_grad():
            out = self._model(input_ids=input_ids, images=[(video_tensor, "video")], output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}
