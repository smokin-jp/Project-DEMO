"""The optional trained models, bundled so the judge can use whichever are installed (each one is simply skipped when missing)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .b32_clip import B32Embedder
from .kick_clf import MODEL_PATH as KICK_PATH, KickClassifier
from .punch_gru import CHECKPOINT as PUNCH_PATH, PunchTypeModel

log = logging.getLogger("referee")


@dataclass
class Extras:
    kick: Optional[KickClassifier] = None       # trained "is this a kick?" detector (CLIP B/32 + logistic regression)
    punch: Optional[PunchTypeModel] = None      # trained punch-type GRU (+ optional CLIP B/32 blend)
    w_kick: float = float(os.getenv("KICK_MODEL_WEIGHT", "0.5"))                         # weight of the trained kick detector against pose geometry + zero-shot CLIP
    kick_crop: str = os.getenv("KICK_CROP", "fighters")                 # "fighters": crop around both fighters first; "frame": the whole frame, as the CLIP preprocessor sees it
    kick_frames: int = 5                        # frames around the impact scored (and smoothed over)

    @property
    def names(self) -> dict:
        return {"kick_model": self.kick is not None, "punch_model": self.punch is not None}


def load_extras(device: str = "cpu") -> Extras:
    """Load whatever trained models are on disk. A missing or broken file switches only that feature off."""
    ex = Extras()
    embedder = None
    if KICK_PATH.exists() or PUNCH_PATH.exists():
        try:
            embedder = B32Embedder(device=device).load()
        except Exception as e:  # no ViT-B/32 weights (offline): the punch GRU still works without the CLIP blend
            log.warning("CLIP ViT-B/32 unavailable (%s): the trained kick detector is off", e)
    if KICK_PATH.exists() and embedder is not None:
        try:
            ex.kick = KickClassifier.load(embedder, KICK_PATH)
            log.warning("trained kick detector loaded")
        except Exception as e:
            log.warning("kick detector not loaded: %s", e)
    if PUNCH_PATH.exists():
        try:
            ex.punch = PunchTypeModel.load(PUNCH_PATH, embedder)
            log.warning("punch-type model loaded%s", " (with CLIP blend)" if embedder else " (skeleton only)")
        except Exception as e:
            log.warning("punch-type model not loaded: %s", e)
    return ex


def fighters_crop(img: Image.Image, pts_norm: list[tuple[float, float]], margin: float = 1.3, min_frac: float = 0.45) -> Image.Image:
    """Square crop around the given points (image fractions), `margin` times their extent and at least `min_frac` of the frame
    height, so the fighters fill the picture the way a photo in a training set would. No points: the frame itself."""
    if not pts_norm:
        return img
    W, H = img.size
    xs = np.array([p[0] for p in pts_norm]) * W
    ys = np.array([p[1] for p in pts_norm]) * H
    side = max(margin * max(xs.max() - xs.min(), ys.max() - ys.min()), min_frac * H)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = side / 2
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
