"""CLIP ViT-B/32 image / text embeddings (through HuggingFace transformers, same weights as OpenAI's `clip` package).

The two models the team trained (the kick classifier and the boxing punch-type fusion) were built on ViT-B/32 embeddings,
whereas the main judge uses ViT-B/16, so they need their own encoder. Loaded lazily and shared between both.
"""
from __future__ import annotations

import threading

import numpy as np
from PIL import Image

B32_NAME = "openai/clip-vit-base-patch32"


class B32Embedder:
    def __init__(self, name: str = B32_NAME, device: str = "cpu"):
        self.name, self.device = name, device
        self._model = None
        self._proc = None
        self._lock = threading.Lock()

    def load(self) -> "B32Embedder":
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_grad_enabled(False)
        self._model = CLIPModel.from_pretrained(self.name).to(self.device).eval()
        self._proc = CLIPProcessor.from_pretrained(self.name)
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _tensor(out):
        import torch

        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        return out[0]

    def images(self, images: list[Image.Image]) -> np.ndarray:
        """(n, 512) L2-normalised image embeddings (same resize / centre-crop / normalisation as `clip.load(...)[1]`)"""
        import torch

        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        with self._lock, torch.no_grad():  # no_grad per call: grad mode is per thread and the API runs requests in worker threads
            inp = self._proc(images=images, return_tensors="pt").to(self.device)
            f = torch.nn.functional.normalize(self._tensor(self._model.get_image_features(**inp)).float(), dim=-1)
        return f.cpu().numpy()

    def texts(self, prompts: list[str]) -> np.ndarray:
        """(n, 512) L2-normalised text embeddings"""
        import torch

        with self._lock, torch.no_grad():
            inp = self._proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            f = torch.nn.functional.normalize(self._tensor(self._model.get_text_features(**inp)).float(), dim=-1)
        return f.cpu().numpy()
