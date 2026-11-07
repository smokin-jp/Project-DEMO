"""OpenAI CLIP zero-shot scorer (official OpenAI checkpoints via HuggingFace transformers)."""
from __future__ import annotations

import threading
from typing import Protocol

from PIL import Image


class Scorer(Protocol):
    name: str

    def embed(self, images: list[Image.Image]): ...

    def score(self, feats, idx: list[int], prompt_set: dict[str, list[str]]) -> dict[str, float]: ...

    def classify(self, images: list[Image.Image], prompt_set: dict[str, list[str]]) -> dict[str, float]: ...


class ClipScorer:
    """label -> probability, averaged over the given images.

    Per label the prompt embeddings are averaged (prompt ensembling); probabilities are a softmax over labels of
    CLIP's scaled cosine similarity. These are *uncalibrated* zero-shot scores, which is why judge.py fuses them
    with pose geometry rather than trusting them alone.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.name = model_name
        self.device = device
        self._model = None
        self._proc = None
        self._text_cache: dict[int, tuple[list[str], object]] = {}
        self._lock = threading.Lock()

    def load(self):
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

    def _image_feats(self, images):
        import torch

        inp = self._proc(images=images, return_tensors="pt").to(self.device)
        f = self._tensor(self._model.get_image_features(**inp))
        return torch.nn.functional.normalize(f, dim=-1)

    def _text_feats(self, prompt_set):
        import torch

        key = id(prompt_set)
        if key in self._text_cache:
            return self._text_cache[key]
        labels, rows = [], []
        for label, prompts in prompt_set.items():
            inp = self._proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            f = torch.nn.functional.normalize(self._tensor(self._model.get_text_features(**inp)), dim=-1)
            rows.append(torch.nn.functional.normalize(f.mean(0), dim=-1))
            labels.append(label)
        out = (labels, torch.stack(rows))
        self._text_cache[key] = out
        return out

    # --- encode once, score many: each crop is embedded a single time and reused for every question
    def embed(self, images):
        with self._lock:
            return self._image_feats(images)

    def score(self, feats, idx, prompt_set):
        """label -> probability, averaged over the embeddings feats[idx]."""
        if not idx:
            return {k: 1.0 / len(prompt_set) for k in prompt_set}
        import torch

        with self._lock, torch.no_grad():  # grad mode is per thread; the API runs requests in worker threads
            labels, txt = self._text_feats(prompt_set)
            scale = self._model.logit_scale.exp()
            probs = (scale * feats[idx] @ txt.T).softmax(dim=-1).mean(0)
        return {l: float(p) for l, p in zip(labels, probs)}

    def classify(self, images, prompt_set):
        if not images:
            return {k: 1.0 / len(prompt_set) for k in prompt_set}
        return self.score(self.embed(images), list(range(len(images))), prompt_set)


class UniformScorer:
    """No-op scorer for tests / when the model is unavailable."""

    name = "uniform"

    def embed(self, images):
        return None

    def score(self, feats, idx, prompt_set):
        return {k: 1.0 / len(prompt_set) for k in prompt_set}

    def classify(self, images, prompt_set):
        return {k: 1.0 / len(prompt_set) for k in prompt_set}
