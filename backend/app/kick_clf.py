"""Trained kick detector: CLIP ViT-B/32 image embedding -> logistic regression (models/kick_detector.joblib, trained by a teammate
on a kick / no-kick image dataset). Answers "does this picture show a kick being delivered?" with a probability and a tuned threshold.

In the referee it is a second opinion on *is this really a kick (and not a step)*, fused with the pose geometry and CLIP's zero-shot
read (judge.py). Per strike, the frames around the impact are scored and a rolling window (teammate's `TemporalKickSmoother`: at least
3 positive frames of the last 5) makes the call, so one odd frame cannot decide.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .temporal_smoothing import TemporalKickSmoother

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "kick_detector.joblib"


class KickClassifier:
    def __init__(self, classifier, threshold: float, embedder):
        self.classifier, self.threshold, self.embedder = classifier, float(threshold), embedder
        self.smoother = TemporalKickSmoother(window_size=5, required_kicks=3)

    @classmethod
    def load(cls, embedder, path: Path | str = MODEL_PATH) -> "KickClassifier":
        import joblib

        saved = joblib.load(path)
        return cls(saved["classifier"], saved["threshold"], embedder)

    def scores(self, images: list[Image.Image]) -> list[float]:
        """probability that each picture shows a kick"""
        if not images:
            return []
        emb = self.embedder.images(images)
        return [float(p) for p in self.classifier.predict_proba(emb)[:, 1]]

    def judge_frames(self, images: list[Image.Image]) -> dict:
        """Score the pictures of one strike in time order and let the rolling window decide."""
        s = self.scores(images)
        sid = "strike"
        self.smoother.reset(sid)
        out = {"kick_detected": False, "positive_frames": 0, "frames_considered": 0}
        for x in s:
            out = self.smoother.update(sid, x >= self.threshold)
        self.smoother.reset(sid)
        return {
            "scores": [round(x, 3) for x in s],
            "mean": round(float(np.mean(s)), 3) if s else None,
            "peak": round(float(max(s)), 3) if s else None,
            "threshold": round(self.threshold, 3),
            "positive_frames": out["positive_frames"],
            "frames_considered": out["frames_considered"],
            "detected": bool(out["kick_detected"]),
        }

    def predict_image(self, image_path: str) -> dict:
        """single picture, same output as the original predict_kick.py"""
        score = self.scores([Image.open(image_path).convert("RGB")])[0]
        detected = score >= self.threshold
        return {"kick_detected": detected, "confidence": round(score, 4), "threshold": self.threshold,
                "decision": "FOUL" if detected else "NO FOUL"}
