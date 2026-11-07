from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Frame(BaseModel):
    t_ms: float
    jpeg_b64: str
    # {"a": {"0": [x, y, visibility], "11": [...]}, "b": {...} | None}; x,y normalised 0..1 (image fractions)
    poses: dict[str, Optional[dict[str, list[float]]]] = Field(default_factory=dict)


class Evidence(BaseModel):
    """What the in-browser pose tracker saw for this candidate strike."""

    thrower: Literal["a", "b"]
    hand: Literal["left", "right"]            # side of the striking limb (hand for punches, leg for kicks/knees)
    strike: Literal["punch", "kick", "knee"] = "punch"
    peak_t_ms: float
    speed_mps: float = 0.0
    extension_m: float = 0.0
    geo_hint: Optional[dict[str, Any]] = None
    # body zones that are legal targets for this strike under the ruleset in force
    legal_zones: list[str] = Field(default_factory=lambda: ["head", "torso"])


class JudgeRequest(BaseModel):
    frames: list[Frame] = Field(min_length=3)
    evidence: Evidence
    return_crops: bool = True
    camera: str = "cam1"            # label of the camera this clip came from (multi-camera judging)


class MultiJudgeRequest(BaseModel):
    """One strike seen by several cameras at the same moment (clips share one clock)."""

    views: list[JudgeRequest] = Field(min_length=1, max_length=4)
    return_crops: bool = True


class Criterion(BaseModel):
    value: Any
    confidence: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Verdict(BaseModel):
    strike: str = "punch"           # punch | kick | knee
    # for kicks/knees: is it really a strike (not a step / lifted leg)? None for punches
    strike_check: Optional[Criterion] = None
    # the seven questions, in order
    thrower: Criterion              # 1. Who threw it?
    intended_target: Criterion      # 2. Who was the intended target?
    contact: Criterion              # 3. Did it make contact?
    contact_location: Criterion     # 4. Where did it make contact?
    legal_area: Criterion           # 5. Legal scoring area?
    legal_glove_part: Criterion     # 6. Legal part of the glove / legal striking surface?
    genuine_punch: Criterion        # 7. Genuine scoring strike (not block / miss / push / glancing)?
    # overall
    scored: bool
    review: bool                    # borderline: hand to the human referee
    confidence: float
    foul: Optional[str] = None      # foul id (matches src/rules/*.js) if the contact was illegal
    summary: str
    latency_ms: float = 0.0
    model: str = ""
    crops: dict[str, str] = Field(default_factory=dict)  # name -> base64 JPEG (what CLIP looked at); "cam2_impact" etc. with several cameras
    # per-camera summary when several views were fused: [{camera, quality, available, p_contact, location, genuine}]
    views: list[dict[str, Any]] = Field(default_factory=list)


class ThrowTypeRequest(BaseModel):
    """The frames around a throw-like fall, in time order (the browser sends up to ~16 of them; 8 evenly spaced are used)."""

    frames: list[Frame] = Field(min_length=3)


class ThrowTypeResult(BaseModel):
    key: str
    technique: str                    # e.g. "Uchi-mata"
    confidence: float                 # probability of the top class (there is no "not a throw" class)
    probs: dict[str, float]
    model_info: dict[str, Any] = Field(default_factory=dict)
