"""VideoChat-Flash — OpenGVLab/VideoChat-Flash-Qwen2-7B_res448.

Requires transformers==4.40.1 — installed in its per-model venv.
Auto-clones the VideoChat-Flash repository on first use.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from models.base import VideoModel, HF_IDS, CACHE, _block_hooks, _remove_hooks


def _ensure_videochat_flash() -> Path:
    repo = CACHE / "VideoChat-Flash"
    if not (repo / "README.md").exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        import shutil
        if repo.exists():
            shutil.rmtree(repo)
        print("Cloning VideoChat-Flash repository ...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/OpenGVLab/VideoChat-Flash.git", str(repo)],
            check=True,
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


class VideoChatFlash(VideoModel):
    name = "videochat-flash"
    n_llm_layers = 28   # Qwen2-7B
    n_enc_layers = 23   # UMT-L (InternVideo2) encoder

    def __init__(self, n_frames: int = 8):
        from transformers import AutoModel, AutoTokenizer
        _ensure_videochat_flash()
        hf = HF_IDS[self.name]
        self.n = n_frames
        self._tok   = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            hf, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()

    def _chat(self, video_path: Path, prompt: str, max_new_tokens: int = 64) -> str:
        return self._model.chat(
            str(video_path), self._tok, prompt,
            return_history=False,
            max_num_frames=self.n,
            generation_config=dict(max_new_tokens=max_new_tokens, do_sample=False),
        )

    def ask(self, video_path: Path, prompt: str) -> str:
        return self._chat(video_path, prompt)

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        llm_blocks = list(self._model.model.layers)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for l in layers:
            def _hook(_, _i, out, _l=l):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 1:
                    captured[_l] = h.detach().float().cpu()
            handles.append(llm_blocks[l].register_forward_hook(_hook))
        with torch.no_grad():
            self._chat(video_path, "Describe the video.", max_new_tokens=1)
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=1) for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        vt = self._model.get_model().get_vision_tower()
        blocks = list(vt.vision_tower.encoder.blocks)
        captured, handles = _block_hooks(blocks, layers)
        with torch.no_grad():
            self._chat(video_path, "Describe the video.", max_new_tokens=1)
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=0, keepdim=True) for l in layers}
