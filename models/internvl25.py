"""InternVL2.5 — OpenGVLab/InternVL2_5-8B (InternViT-300M + InternLM2-8B).

Requires transformers~=4.45 — the remote model code uses the pre-4.50 API.
The run script pins the correct version in .venv-internvl2.5.
"""

from pathlib import Path
import torch
from models.base import VideoModel, HF_IDS, load_frames, _remove_hooks


class InternVL25(VideoModel):
    name = "internvl2.5"
    n_llm_layers = 32   # InternLM2-8B
    n_enc_layers = 24   # InternViT-300M

    def __init__(self, n_frames: int = 8):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer

        hf = HF_IDS[self.name]
        self.n = n_frames
        self._tok   = AutoTokenizer.from_pretrained(hf, trust_remote_code=True, use_fast=False)
        self._model = AutoModel.from_pretrained(
            hf, trust_remote_code=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, device_map={"": 0},
        ).eval()

        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _to_pixel_values(self, video_path: Path) -> torch.Tensor:
        from PIL import Image
        frames = load_frames(video_path, self.n)
        pil    = [Image.fromarray(f) for f in frames]
        pixels = torch.stack([self._transform(f) for f in pil])
        return pixels.to(self._model.device, torch.bfloat16)

    def ask(self, video_path: Path, prompt: str) -> str:
        pv          = self._to_pixel_values(video_path)
        num_patches = [1] * pv.shape[0]
        prefix      = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches))])
        response, _ = self._model.chat(
            self._tok, pv, prefix + prompt,
            dict(max_new_tokens=64, do_sample=False),
            num_patches_list=num_patches, history=None, return_history=True,
        )
        return response

    def extract_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        pv          = self._to_pixel_values(video_path)
        num_patches = [1] * pv.shape[0]
        prefix      = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches))])
        llm_blocks = list(self._model.language_model.model.layers)
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for l in layers:
            def _hook(_, _i, out, _l=l):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 1:
                    captured[_l] = h.detach().float().cpu()
            handles.append(llm_blocks[l].register_forward_hook(_hook))
        with torch.no_grad():
            self._model.chat(
                self._tok, pv, prefix + "Describe the video.",
                dict(max_new_tokens=1, do_sample=False),
                num_patches_list=num_patches, history=None, return_history=True,
            )
        _remove_hooks(handles)
        return {l: captured[l].mean(dim=1) for l in layers}

    def extract_vision_features(self, video_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
        pv = self._to_pixel_values(video_path)
        with torch.no_grad():
            out = self._model.vision_model(pv, output_hidden_states=True)
        return {l: out.hidden_states[l].mean(dim=1).float().cpu() for l in layers}
