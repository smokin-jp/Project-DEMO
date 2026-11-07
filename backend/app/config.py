"""Backend settings (override with environment variables)."""
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # OpenAI CLIP weights (official checkpoints, loaded through HuggingFace transformers).
    # ViT-B/16 is a good CPU default; "openai/clip-vit-large-patch14" is noticeably better but ~4x slower.
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch16")
    device: str = os.getenv("CLIP_DEVICE", "cpu")

    # Weight given to the pose-geometry evidence (rest goes to CLIP) when fusing each criterion.
    # CLIP cannot see "which part of the glove"; geometry cannot see glove orientation. Tune on real footage.
    w_geo: dict = field(
        default_factory=lambda: {"contact": 0.65, "location": 0.70, "glove": 0.35, "genuine": 0.65}
    )

    # Overall decision thresholds on the weakest criterion confidence.
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.55"))   # >= : scoring punch
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.35"))  # [review, score): flag for human referee
    foul_threshold: float = float(os.getenv("FOUL_THRESHOLD", "0.50"))      # confidence needed to call a foul (a clear majority)

    max_frames: int = 24
    crop_thumb_px: int = 160


settings = Settings()
