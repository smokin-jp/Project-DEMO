"""Which judo throw is it?  CLIP frame embeddings -> a small temporal classifier trained on the `adenhaus/judo_throws` clips.

Shared by training (training/03b_train_clip.py) and serving (backend /api/throw_type) so the preprocessing cannot drift.
The 4 classes are exactly those of the dataset: ippon seoi nage, o goshi, osoto gari, uchi mata. There is NO "not a throw" class,
so callers must only ask when something throw-like really happened (a fall next to the opponent).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

N_FRAMES = 8
PRETTY = {"ippon_seoi_nage": "Ippon-seoi-nage", "o_goshi": "O-goshi", "osoto_gari": "Osoto-gari", "uchi_mata": "Uchi-mata"}


def letterbox(img: Image.Image) -> Image.Image:
    """pad to a square instead of centre-cropping, so a fighter at the side of a wide frame is not cut off (same as training)"""
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (114, 114, 114))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def pick_frames(n_available: int, n: int = N_FRAMES) -> list[int]:
    """indices of n evenly spaced frames across the clip (repeats if the clip has fewer)"""
    return np.linspace(0, max(n_available - 1, 0), n).round().astype(int).tolist()


def build_head(n_classes: int, d_in: int = 512, hid: int = 128):
    import torch.nn as nn

    class TemporalHead(nn.Module):
        """per-frame projection -> small temporal conv -> mean+max pooling -> classes (~100 k parameters)"""

        def __init__(self):
            super().__init__()
            self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(0.3), nn.Linear(d_in, hid), nn.GELU())
            self.conv = nn.Sequential(nn.Conv1d(hid, hid, 3, padding=1), nn.GELU(), nn.Dropout(0.3))
            self.out = nn.Sequential(nn.Dropout(0.4), nn.Linear(2 * hid, n_classes))

        def forward(self, x):  # (B, F, D)
            import torch

            h = self.proj(x)
            h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
            return self.out(torch.cat([h.mean(1), h.max(1).values], 1))

    return TemporalHead()


class ThrowTypeModel:
    def __init__(self, net, classes: list[str], meta: dict):
        self.net, self.classes, self.meta = net.eval(), classes, meta

    @classmethod
    def load(cls, path: str):
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = build_head(len(ck["classes"]), ck.get("dim", 512), ck.get("hidden", 128))
        net.load_state_dict(ck["model_state_dict"])
        return cls(net, list(ck["classes"]), ck)

    def predict(self, images: list[Image.Image], scorer) -> dict:
        """images: the frames of the clip in time order (any number >= 1). Embeds N_FRAMES evenly spaced ones with CLIP."""
        import torch

        chosen = [letterbox(images[i]) for i in pick_frames(len(images))]
        feats = scorer.embed(chosen)  # (N_FRAMES, 512), L2-normalised
        with torch.no_grad():
            probs = torch.softmax(self.net(feats[None].float()), dim=1)[0].numpy()
        top = int(probs.argmax())
        key = self.classes[top]
        return {
            "key": key, "technique": PRETTY.get(key, key), "confidence": float(probs[top]),
            "probs": {PRETTY.get(c, c): float(p) for c, p in zip(self.classes, probs)},
        }
