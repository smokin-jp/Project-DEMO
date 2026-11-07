"""Punch TYPE (cross / jab / lead hook / lead uppercut / rear hook / rear uppercut) from the thrower's skeleton.

A bidirectional GRU trained by a teammate on BoxingVI skeleton sequences (boxing_ai/, checkpoint
boxing_ai/checkpoints/skeleton_gru_normalized_velocity_best.pth: 25 frames x 17 COCO keypoints, hip-centred, torso-scaled, position +
velocity). Optionally blended 80 / 20 with a CLIP ViT-B/32 zero-shot read of the picture, as in the teammate's fusion prototype.

HONEST STATUS: the checkpoint's own validation macro-F1 is about 0.43 (held-out videos V1 / V4), so this is an experimental
*label* shown to the referee ("looks like a lead hook, 41 %"); it never changes the scoring. The GRU architecture here is identical to
boxing_ai/fusion_model.py (kept in sync by tests that load the checkpoint).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CHECKPOINT = Path(__file__).resolve().parent.parent.parent / "boxing_ai" / "checkpoints" / "skeleton_gru_normalized_velocity_best.pth"
CLASSES = ["cross", "jab", "lead hook", "lead uppercut", "rear hook", "rear uppercut"]
N_FRAMES = 25
# MediaPipe (33-point) index of each COCO-17 keypoint: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
COCO_FROM_MEDIAPIPE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MIN_VALID_FRAMES = 6  # of the thrower's frames in the clip, otherwise the skeleton is too patchy to read

DESCRIPTIONS = [
    "a straight rear-hand cross punch with the rear shoulder and hip rotating forward",
    "a straight lead-hand jab with the front arm extending forward and rear hand guarding the face",
    "a lead-hand hook with the elbow bent and fist travelling horizontally in an arc",
    "a lead-hand uppercut with the elbow bent and fist travelling upwards from below",
    "a rear-hand hook with the elbow bent and fist travelling horizontally in an arc",
    "a rear-hand uppercut with the elbow bent and fist travelling upwards from below",
]
TEMPLATES = ["a photo of a boxer performing a {}.", "a frame from a boxing match showing a {}.", "a boxer throwing a {}."]


def build_gru():
    import torch.nn as nn

    class BoxingGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=68, hidden_size=96, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
            self.classifier = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.35), nn.Linear(96, len(CLASSES)))

        def forward(self, x, lengths):  # x: (B, 25, 17, 4); lengths: (B,)
            import torch
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            b = x.shape[0]
            packed = pack_padded_sequence(x.reshape(b, N_FRAMES, 68), lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=N_FRAMES)
            mask = (torch.arange(N_FRAMES).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1).to(out.dtype)
            pooled = (out * mask).sum(1) / lengths.unsqueeze(1).to(out.dtype).clamp_min(1.0)
            return self.classifier(pooled)

    return BoxingGRU()


def coco_keypoints(pose: dict | None, aspect: float) -> np.ndarray | None:
    """one frame's thrower pose ({"0": [x, y, v], ...}, image fractions) -> (17, 2) in pixel-shaped units, or None.
    The eyes are not sent by older clients: the nose stands in for them."""
    if not pose:
        return None
    out = np.zeros((17, 2), dtype=np.float32)
    for k, mp_i in enumerate(COCO_FROM_MEDIAPIPE):
        v = pose.get(str(mp_i)) or (pose.get("0") if mp_i in (2, 5) else None)
        if v is None:
            return None
        out[k] = (v[0] * aspect, v[1])
    return out


def resample(times: list[float], frames: list[np.ndarray | None], n: int = N_FRAMES) -> np.ndarray | None:
    """Frames of unequal spacing (some missing) -> exactly n frames evenly spaced from the first to the last valid frame,
    by linear interpolation of every keypoint. None when too few valid frames."""
    pts = [(t, f) for t, f in zip(times, frames) if f is not None]
    if len(pts) < MIN_VALID_FRAMES:
        return None
    t = np.array([p[0] for p in pts], dtype=np.float64)
    order = np.argsort(t)
    t = t[order]
    arr = np.stack([pts[i][1] for i in order])  # (m, 17, 2)
    if t[-1] - t[0] < 1e-6:
        return None
    grid = np.linspace(t[0], t[-1], n)
    out = np.empty((n, 17, 2), dtype=np.float32)
    flat = arr.reshape(len(t), -1)
    for j in range(flat.shape[1]):
        out.reshape(n, -1)[:, j] = np.interp(grid, t, flat[:, j])
    return out


class PunchTypeModel:
    def __init__(self, net, meta: dict, embedder=None):
        self.net = net.eval()
        self.meta = meta
        self.mean = meta["velocity_mean"].cpu().numpy()
        self.std = np.where(np.abs(meta["velocity_std"].cpu().numpy()) < 1e-6, 1.0, meta["velocity_std"].cpu().numpy())
        idx = meta["keypoint_indices"]
        self.ls, self.rs, self.lh, self.rh = idx["left_shoulder"], idx["right_shoulder"], idx["left_hip"], idx["right_hip"]
        self.embedder = embedder
        self._text = None

    @classmethod
    def load(cls, path: Path | str = CHECKPOINT, embedder=None) -> "PunchTypeModel":
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=True)
        if list(ck["classes"]) != CLASSES:
            raise ValueError("checkpoint class order does not match")
        net = build_gru()
        net.load_state_dict(ck["model_state_dict"])
        return cls(net, ck, embedder)

    def features(self, seq: np.ndarray) -> np.ndarray:
        """(25, 17, 2) raw keypoints -> (25, 17, 4): hip-centred, torso-scaled positions + standardised velocities
        (the same recipe as the training script and the teammate's `preprocess_skeleton`)"""
        hip = (seq[:, self.lh] + seq[:, self.rh]) / 2
        sh = (seq[:, self.ls] + seq[:, self.rs]) / 2
        scale = float(np.median(np.linalg.norm(sh - hip, axis=1)))
        if not np.isfinite(scale) or scale < 1e-6:
            raise ValueError("invalid torso length")
        pos = (seq - np.median(hip, axis=0)) / scale
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        vel = (vel - self.mean) / self.std
        vel[0] = 0.0
        return np.concatenate([pos, vel], axis=-1).astype(np.float32)

    def _clip_probs(self, images) -> np.ndarray | None:
        if self.embedder is None or not images:
            return None
        if self._text is None:
            feats = self.embedder.texts([t.format(d) for d in DESCRIPTIONS for t in TEMPLATES])
            cls = feats.reshape(len(DESCRIPTIONS), len(TEMPLATES), -1).mean(1)
            self._text = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        sims = 100.0 * (self.embedder.images(images) @ self._text.T)
        sims = sims.mean(0)
        e = np.exp(sims - sims.max())
        return e / e.sum()

    def predict(self, seq: np.ndarray, images=None, gru_weight: float = 0.8) -> dict:
        import torch

        x = torch.from_numpy(self.features(seq)).unsqueeze(0)
        with torch.no_grad():
            gru = torch.softmax(self.net(x, torch.tensor([N_FRAMES])), dim=1)[0].numpy()
        clip = self._clip_probs(images)
        p = gru if clip is None else gru_weight * gru + (1 - gru_weight) * clip
        k = int(np.argmax(p))
        return {
            "label": CLASSES[k], "confidence": round(float(p[k]), 3),
            "probs": {c: round(float(v), 3) for c, v in zip(CLASSES, p)},
            "gru": {c: round(float(v), 3) for c, v in zip(CLASSES, gru)},
            "clip": None if clip is None else {c: round(float(v), 3) for c, v in zip(CLASSES, clip)},
            "source": "skeleton GRU" + (" + CLIP ViT-B/32 (80/20)" if clip is not None else ""),
            "note": f"experimental: the model's validation macro-F1 was {float(self.meta['best_macro_f1']):.2f}; a label only, it never changes the score",
        }
