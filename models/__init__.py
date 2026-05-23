"""
Model loading and inference for temporal probing experiments.

Each model lives in its own file to isolate dependencies and inference patterns.
"""

from models.base import VideoModel, HF_IDS, load_frames, CACHE

_CLASSES: dict[str, type[VideoModel]] = {}

def _register_lazy():
    """Populate _CLASSES on first call to load_model."""
    if _CLASSES:
        return
    from models.llava_next_video import LLaVANextVideo
    from models.video_llama2 import VideoLLaMA2
    from models.qwen25_vl import Qwen25VL
    from models.molmo2 import Molmo2
    from models.internvl25 import InternVL25
    from models.videollama3 import VideoLLaMA3Adapter
    from models.videochat_flash import VideoChatFlash

    _CLASSES.update({
        "llava-next-video": LLaVANextVideo,
        "video-llama2":     VideoLLaMA2,
        "qwen2.5-vl":       Qwen25VL,
        "molmo2":           Molmo2,
        "internvl2.5":      InternVL25,
        "videollama3":      VideoLLaMA3Adapter,
        "videochat-flash":  VideoChatFlash,
    })


def load_model(name: str) -> VideoModel:
    _register_lazy()
    if name not in _CLASSES:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_CLASSES)}")
    return _CLASSES[name]()
