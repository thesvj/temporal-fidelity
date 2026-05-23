"""Shared base class, video loading, and GPU compatibility patches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# B200 compatibility: cuRAND init.normal_ has no sm_100 PTX kernel.
# Route CUDA random init through CPU — values are immediately overwritten by
# pretrained weights, so this has zero effect on results.
# ---------------------------------------------------------------------------
_orig_init_normal_ = torch.nn.init.normal_

def _cuda_safe_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, **kwargs) -> torch.Tensor:
    if tensor.is_cuda:
        cpu_t = tensor.detach().cpu()
        _orig_init_normal_(cpu_t, mean=mean, std=std, **kwargs)
        tensor.copy_(cpu_t)
        return tensor
    return _orig_init_normal_(tensor, mean=mean, std=std, **kwargs)

torch.nn.init.normal_ = _cuda_safe_normal_

# ---------------------------------------------------------------------------
# Rope compatibility: some models expect a "default" rope type removed in
# newer transformers. Add it back as standard RoPE if missing.
# ---------------------------------------------------------------------------
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_FUNCS
    if "default" not in _ROPE_FUNCS:
        def _default_rope(config, device=None, **_kw):
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            partial = getattr(config, "partial_rotary_factor", 1.0)
            dim = int(head_dim * partial)
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            return inv_freq, 1.0
        _ROPE_FUNCS["default"] = _default_rope
except ImportError:
    pass

# ---------------------------------------------------------------------------
# HuggingFace model IDs
# ---------------------------------------------------------------------------

HF_IDS = {
    "llava-next-video": "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
    "video-llama2":     "DAMO-NLP-SG/VideoLLaMA2-7B",
    "qwen2.5-vl":       "Qwen/Qwen2.5-VL-7B-Instruct",
    "molmo2":           "allenai/Molmo2-8B",
    "internvl2.5":      "OpenGVLab/InternVL2_5-8B",
    "videollama3":      "DAMO-NLP-SG/VideoLLaMA3-7B",
    "videochat-flash":  "OpenGVLab/VideoChat-Flash-Qwen2-7B_res448",
}

CACHE = Path.home() / ".cache" / "temporal_probing"

# ---------------------------------------------------------------------------
# Shared video loading
# ---------------------------------------------------------------------------

def load_frames(path: Path, n: int = 8) -> np.ndarray:
    """Sample n frames uniformly from video -> (n, H, W, 3) uint8."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
    total = len(vr)
    idx = [int(i * total / n) for i in range(n)]
    return vr.get_batch(idx).asnumpy()


# ---------------------------------------------------------------------------
# Shared utility: forward hooks to capture encoder block outputs
# ---------------------------------------------------------------------------

def _block_hooks(blocks, layers: list[int]) -> tuple[dict, list]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for l in layers:
        def _hook(_, _i, out, _l=l):
            h = out[0] if isinstance(out, tuple) else out
            captured[_l] = h.detach().float().cpu()
        handles.append(blocks[l].register_forward_hook(_hook))
    return captured, handles


def _remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class VideoModel(ABC):
    name: str
    n_llm_layers: int = 32
    n_enc_layers: int = 24

    @abstractmethod
    def ask(self, video_path: Path, prompt: str) -> str:
        """Run video + text prompt, return model's text response."""

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        raise NotImplementedError(f"{self.name}: LLM feature extraction not implemented")

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        raise NotImplementedError(f"{self.name}: vision encoder feature extraction not implemented")
