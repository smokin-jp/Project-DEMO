"""Answers the seven strike questions by fusing pose geometry with CLIP zero-shot scores on image crops.
Works for punches, kicks and knee strikes, from one camera or several.

 0 Is it a real kick/knee?  (kick/knee only) pose: speed + leg lift  x  CLIP "kick vs step / stance" on the crops
 1 Who threw it?            pose: whose limb moved fast towards the other fighter (browser tracker + geometry check)
 2 Who was the target?      pose: the other fighter in the frame (None if nobody to hit)
 3 Did it make contact?     geometry (striking surface vs opponent zones) x CLIP "contact vs miss" on impact crops
 4 Where?                   geometry zone softmin (+ facing => back) x CLIP location prompts
 5 Legal scoring area?      location in the ruleset's legal zones (head / torso always; legs only if the rules allow)
 6 Legal glove part / surface?  punch: knuckle part of the glove; kick: foot or shin; knee: knee
 7 Genuine strike?          clean vs block / miss / push / glancing: guard, speed, recoil, strike line  x  CLIP

Several cameras: every view is analysed on its own (`analyze_view`) and gets a quality weight (joint visibility and
foreshortening: a strike seen side-on is believed over one pointing at the lens). `decide` then fuses the per-view
distributions with a quality-weighted geometric mean, and a clear miss from a good-quality view vetoes "contact".
One camera is simply the one-view case.
"""
from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from .clip_model import Scorer
from .config import Settings, settings as default_settings
from .geometry import LEGAL_SURFACE, GeoResult, analyze, build_person, normalise
from .prompts import PROMPT_SETS
from .schemas import Criterion, Evidence, JudgeRequest, MultiJudgeRequest, Verdict

NAMES = {"a": "RED", "b": "BLUE"}
WORD = {"punch": "punch", "kick": "kick", "knee": "knee strike"}


def decode_jpeg(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def encode_jpeg(img: Image.Image, px: int) -> str:
    im = img.copy()
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def crop_square(img: Image.Image, cx_norm: float, cy_norm: float, size_px: float) -> Image.Image:
    """Square crop centred on a normalised point; PIL pads out-of-frame areas with black."""
    W, H = img.size
    cx, cy, half = cx_norm * W, cy_norm * H, size_px / 2
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def fuse(geo: Optional[dict[str, float]], clip: Optional[dict[str, float]], w_geo: float) -> dict[str, float]:
    """Weighted geometric mean of the two distributions (a soft AND: both must agree for high probability)."""
    if geo is None and clip is None:
        raise ValueError("nothing to fuse")
    if geo is None:
        return normalise(dict(clip))
    if clip is None:
        return normalise(dict(geo))
    out = {k: (max(geo[k], 1e-4) ** w_geo) * (max(clip[k], 1e-4) ** (1 - w_geo)) for k in geo}
    return normalise(out)


def fuse_views(dists: list[dict[str, float]], weights: list[float]) -> dict[str, float]:
    """Quality-weighted geometric mean across camera views. One view returns itself unchanged."""
    tot = sum(weights) or 1.0
    out = {k: math.exp(sum(w * math.log(max(d[k], 1e-4)) for d, w in zip(dists, weights)) / tot) for k in dists[0]}
    return normalise(out)


def _argmax(d: dict[str, float]) -> tuple[str, float]:
    k = max(d, key=d.get)
    return k, d[k]


def _r(d: Optional[dict[str, float]]) -> Optional[dict[str, float]]:
    return None if d is None else {k: round(v, 3) for k, v in d.items()}


@dataclass
class ViewResult:
    camera: str
    ev: Evidence
    geo: GeoResult
    peak_img: Image.Image
    crops: dict[str, Image.Image] = field(default_factory=dict)
    clip: dict[str, Optional[dict[str, float]]] = field(default_factory=dict)
    dist: Optional[dict[str, Optional[dict[str, float]]]] = None  # this view's fused geometry x CLIP distributions
    quality: float = 0.05


def analyze_view(req: JudgeRequest, scorer: Scorer, cfg: Settings = default_settings) -> ViewResult:
    """Geometry + CLIP for ONE camera's clip."""
    ev = req.evidence
    strike = ev.strike
    prompts = PROMPT_SETS[strike]
    frames = sorted(req.frames[: cfg.max_frames], key=lambda f: f.t_ms)
    images = [decode_jpeg(f.jpeg_b64) for f in frames]
    W, H = images[0].size
    aspect = W / H
    times = [f.t_ms for f in frames]
    people = [{k: build_person(f.poses.get(k), aspect) for k in ("a", "b")} for f in frames]

    geo: GeoResult = analyze(times, people, aspect, ev.thrower, ev.hand, ev.peak_t_ms, ev.speed_mps, strike)
    p = geo.peak_index
    target_id = geo.target
    vr = ViewResult(camera=req.camera, ev=ev, geo=geo, peak_img=images[p], quality=geo.quality)
    clip: dict[str, Optional[dict[str, float]]] = {"contact": None, "location": None, "glove": None, "genuine": None, "check": None}

    if geo.available:
        op = people[p].get(target_id) or next((people[q][target_id] for q in range(len(frames)) if people[q].get(target_id)), None)
        torso_px = (op.torso if op else 0.25) * H
        lim = min(W, H)
        k = 2.8 if strike == "kick" else 2.4
        s_imp = float(np.clip(k * torso_px, 160, lim))
        s_weapon = float(np.clip((1.3 if strike == "kick" else 1.1) * torso_px, 128, lim))
        cx, cy = geo.impact_norm
        impact = [crop_square(images[q], cx, cy, s_imp) for q in range(max(0, p - 1), min(len(frames), p + 2))]
        vr.crops["impact"] = impact[len(impact) // 2]
        vr.crops["weapon"] = crop_square(images[p], cx, cy, s_weapon)
        q_after = min(len(frames) - 1, p + 2)
        opq = people[q_after].get(target_id)
        react = []
        if opq is not None:
            hx, hy = float(opq.head_c[0] / aspect), float(opq.head_c[1])
            vr.crops["reaction"] = crop_square(images[q_after], hx, hy, s_imp)
            react = [vr.crops["reaction"]]
        # embed every crop once, then ask all questions against the same embeddings
        feats = scorer.embed(impact + react + [vr.crops["weapon"]])
        n_i, n_r = len(impact), len(react)
        i_idx = list(range(n_i))
        r_idx = list(range(n_i, n_i + n_r))
        g_idx = n_i + n_r
        c = scorer.score(feats, i_idx + r_idx, prompts["contact"])
        clip["contact"] = {"contact": c["contact"], "no_contact": c["no_contact"]}
        clip["location"] = scorer.score(feats, i_idx, prompts["location"])
        clip["glove"] = scorer.score(feats, [g_idx, i_idx[n_i // 2]], prompts["glove"])
        clip["genuine"] = scorer.score(feats, i_idx + r_idx, prompts["genuine"])
        if "check" in prompts:
            clip["check"] = scorer.score(feats, [g_idx, i_idx[n_i // 2]], prompts["check"])
    vr.clip = clip

    w = cfg.w_geo
    gp = geo.strike_p if geo.strike_p is not None else 0.5
    vr.dist = {
        "check": fuse({"real": gp, "not_real": 1 - gp}, clip["check"], 0.6) if strike != "punch" else None,
    }
    if geo.available:
        vr.dist["contact"] = fuse({"contact": geo.contact_p, "no_contact": 1 - geo.contact_p}, clip["contact"], w["contact"])
        vr.dist["location"] = fuse(geo.location_p, clip["location"], w["location"])
        vr.dist["glove"] = fuse(geo.glove_p, clip["glove"], w["glove"] if strike == "punch" else 0.45)
        vr.dist["genuine"] = fuse(geo.genuine_p, clip["genuine"], w["genuine"])
    return vr


def _view_summary(v: ViewResult) -> dict:
    d = v.dist or {}
    out = {"camera": v.camera, "quality": round(v.quality, 3), "available": v.geo.available,
           "visibility": v.geo.metrics.get("visibility"), "foreshortening": v.geo.metrics.get("foreshortening")}
    if v.geo.available and d.get("contact"):
        out.update(p_contact=round(d["contact"]["contact"], 3), location=_argmax(d["location"])[0],
                   genuine=_argmax(d["genuine"])[0])
    return out


def decide(views: list[ViewResult], scorer_name: str, return_crops: bool, t0: float, cfg: Settings = default_settings) -> Verdict:
    """One verdict from one or more analysed camera views."""
    first = views[0]
    ev = first.ev
    strike = ev.strike
    word = WORD[strike]
    avail = [v for v in views if v.geo.available]
    thrower_name = NAMES[ev.thrower]
    multi = len(views) > 1
    summaries = [_view_summary(v) for v in views] if multi else []

    def thumbs() -> dict[str, str]:
        if not return_crops:
            return {}
        out = {}
        for v in views:
            shots = {"frame": v.peak_img, **v.crops}  # full-frame screenshot at the moment of impact
            for k2, img in shots.items():
                out[f"{v.camera}_{k2}" if multi else k2] = encode_jpeg(img, 360 if k2 == "frame" else cfg.crop_thumb_px)
        return out

    # ---- thrower / target (quality-weighted over the views that saw both fighters)
    base_views = avail or views
    ws = [v.quality for v in base_views]
    thrower_p = float(sum(v.geo.thrower_p * w_ for v, w_ in zip(base_views, ws)) / (sum(ws) or 1.0))
    target_id = avail[0].geo.target if avail else None
    q1 = Criterion(value=ev.thrower, confidence=round(thrower_p, 3),
                   detail={"name": thrower_name, "hand": ev.hand, "strike": strike, "limb_speed_mps": round(ev.speed_mps, 2),
                           "extension_m": round(ev.extension_m, 2), "cameras": len(views)})
    q2 = Criterion(value=target_id, confidence=0.9,
                   detail={"name": NAMES.get(target_id) if target_id else None,
                           "note": None if target_id else first.geo.metrics.get("reason", "no opponent visible")})

    # ---- 0 (kick / knee): is it really a strike?
    strike_check = None
    if strike != "punch":
        chk = [v.dist["check"] for v in base_views if v.dist and v.dist.get("check")]
        f_chk = fuse_views(chk, [v.quality for v in base_views]) if chk else {"real": 0.5, "not_real": 0.5}
        gps = [v.geo.strike_p for v in base_views if v.geo.strike_p is not None]
        strike_check = Criterion(value=f_chk["real"] >= 0.5, confidence=round(max(f_chk["real"], 1 - f_chk["real"]), 3),
                                 detail={"p_real": round(f_chk["real"], 3), "geometry": round(float(np.mean(gps)) if gps else 0.5, 3),
                                         "clip": _r(first.clip.get("check")),
                                         "note": "kinematics (speed, leg lift) x CLIP kick-vs-step"})
    confirmed = strike_check is None or bool(strike_check.value)

    if not avail:
        no = Criterion(value=False, confidence=0.9, detail={"note": first.geo.metrics.get("reason", "no target in frame")})
        na = Criterion(value=None, confidence=0.0, detail={"note": "n/a: no contact"})
        return Verdict(
            strike=strike, strike_check=strike_check,
            thrower=q1, intended_target=q2, contact=no, contact_location=Criterion(value="none", confidence=0.9),
            legal_area=na, legal_glove_part=na, genuine_punch=Criterion(value=False, confidence=0.9, detail={"classification": "miss"}),
            scored=False, review=False, confidence=0.0,
            summary=f"{thrower_name} {ev.hand} {word}: no opponent in frame, nothing to score.",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1), model=scorer_name, crops=thumbs(), views=summaries,
        )

    # ---- fuse the per-view distributions (quality-weighted)
    weights = [v.quality for v in avail]
    fused = {k: fuse_views([v.dist[k] for v in avail], weights) for k in ("contact", "location", "glove", "genuine")}
    f_contact, f_loc, f_glove, f_gen = fused["contact"], fused["location"], fused["glove"], fused["genuine"]

    # a clear "no contact" from a good-quality view vetoes contact (one camera may see a punch pass in front of the
    # face that another camera sees as overlapping it)
    veto = any(v.quality >= 0.5 and v.dist["contact"]["contact"] < 0.30 for v in avail) and len(avail) > 1
    contact_p = min(f_contact["contact"], 0.45) if veto else f_contact["contact"]
    contact_p_geo = float(np.average([v.geo.contact_p for v in avail], weights=weights))

    q3 = Criterion(value=contact_p >= 0.5, confidence=round(max(contact_p, 1 - contact_p), 3),
                   detail={"p_contact": round(contact_p, 3), "geometry": round(contact_p_geo, 3), "clip": _r(first.clip.get("contact")),
                           "veto_by_good_view": bool(veto)})

    # ---- 4 location
    f_loc = dict(f_loc)
    if strike == "punch":  # a punch to the thigh or knee is just "below the belt"
        f_loc["below_belt"] += f_loc["legs"] + f_loc["knee_joint"]
        f_loc["legs"] = f_loc["knee_joint"] = 0.0
    loc, loc_p = _argmax(f_loc)
    q4 = Criterion(value=loc if contact_p >= 0.5 else "none", confidence=round(loc_p if contact_p >= 0.5 else 1 - contact_p, 3),
                   detail={"distribution": _r(f_loc), "geometry": _r(avail[0].geo.location_p), "clip": _r(first.clip.get("location"))})

    # ---- 5 legal scoring area under the ruleset in force (punches can never legally score on the legs)
    legal = set(ev.legal_zones)
    if strike == "punch":
        legal &= {"head", "torso"}
    p_legal = sum(f_loc[z] for z in legal if z in f_loc)
    q5 = Criterion(value=(p_legal >= 0.5) if contact_p >= 0.5 else None, confidence=round(max(p_legal, 1 - p_legal), 3),
                   detail={"p_legal": round(p_legal, 3), "legal_zones": sorted(legal)})

    # ---- 6 legal glove part / striking surface
    part, _ = _argmax(f_glove)
    p_legal_part = f_glove[LEGAL_SURFACE[strike]]
    q6 = Criterion(value=(p_legal_part >= 0.5) if contact_p >= 0.5 else None, confidence=round(max(p_legal_part, 1 - p_legal_part), 3),
                   detail={"part": part, "distribution": _r(f_glove), "geometry": _r(avail[0].geo.glove_p), "clip": _r(first.clip.get("glove")),
                           "note": "CLIP is weak at glove orientation; treat as low-confidence evidence"})

    # ---- 7 genuine strike
    kind, kind_p = _argmax(f_gen)
    if veto and kind == "clean":  # a vetoed contact cannot be a clean strike
        kind, kind_p = "miss", max(f_gen["miss"], 0.5)
    p_clean = f_gen["clean"]
    gm = avail[0].geo.metrics
    q7 = Criterion(value=kind == "clean", confidence=round(p_clean if kind == "clean" else kind_p, 3),
                   detail={"classification": kind, "distribution": _r(f_gen), "geometry": _r(avail[0].geo.genuine_p), "clip": _r(first.clip.get("genuine")),
                           **{k2: v2 for k2, v2 in gm.items() if k2 in ("guard_dist_torso", "recoil_term", "perp_ratio", "speed_mps")}})

    # ---- overall
    in_contact = contact_p >= 0.5
    values_ok = confirmed and in_contact and p_legal >= 0.5 and p_legal_part >= 0.5 and kind == "clean"
    overall = math.exp(np.mean(np.log([max(contact_p, 1e-3), max(p_legal, 1e-3), max(p_legal_part, 1e-3), max(p_clean, 1e-3)])))
    scored = bool(values_ok and overall >= cfg.score_threshold)
    near_clean = confirmed and in_contact and p_legal >= 0.5 and p_legal_part >= 0.5 and p_clean >= 0.30
    review = bool(not scored and overall >= cfg.review_threshold and (values_ok or near_clean))

    foul = None
    if confirmed and in_contact and f_gen["miss"] < 0.5 and kind != "block":
        # (probability, foul id, location label or None for surface-based fouls)
        cands = [(f_loc["below_belt"], "low_blow", "below_belt"), (f_loc["back"], "back_head", "back")]
        if strike == "punch":
            cands += [(f_glove["open_glove"], "open_glove", None), (f_glove["forearm_or_elbow"], "elbow", None)]
        else:
            cands.append((f_loc["knee_joint"], "knee_joint_kick" if strike == "kick" else "low_blow", "knee_joint"))
            if strike == "kick" and "legs" not in legal:
                cands.append((f_loc["legs"], "kick_below_waist", "legs"))
        # a foul needs a clear majority, or to be the top-ranked location with >= 40% (2D skeletons put the groin,
        # thigh and knee within a few cm of each other, so their probability mass is naturally split)
        for pr, fid, zone in sorted(cands, key=lambda c: -c[0]):
            if pr >= cfg.foul_threshold or (zone is not None and loc == zone and pr >= 0.40):
                foul = fid
                break

    who = f"{thrower_name} {ev.hand} {word}"
    tgt = NAMES[target_id]
    cams = f" [{len(avail)} cameras]" if len(avail) > 1 else ""
    if not confirmed:
        summary = f"{who}: not a real {word} (step / raised leg), ignored."
    elif foul:
        summary = f"{who} -> {tgt}: contact, but illegal ({foul.replace('_', ' ')}){cams}."
    elif scored:
        summary = f"{who} -> {tgt}: clean {word} to the {loc.replace('_', ' ')} ({overall:.0%}){cams}."
    elif review:
        summary = f"{who} -> {tgt}: borderline ({kind}, {overall:.0%}): referee review{cams}."
    elif not in_contact:
        summary = f"{who} -> {tgt}: no contact (miss){' as seen from the best camera' if veto else ''}{cams}."
    else:
        summary = f"{who} -> {tgt}: contact but not a scoring {word} ({kind}){cams}."

    return Verdict(
        strike=strike, strike_check=strike_check,
        thrower=q1, intended_target=q2, contact=q3, contact_location=q4, legal_area=q5, legal_glove_part=q6, genuine_punch=q7,
        scored=scored, review=review, confidence=round(float(overall), 3), foul=foul, summary=summary,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1), model=scorer_name, crops=thumbs(), views=summaries,
    )


def judge(req: JudgeRequest, scorer: Scorer, cfg: Settings = default_settings) -> Verdict:
    """Single-camera judgement (the one-view case)."""
    t0 = time.perf_counter()
    view = analyze_view(req, scorer, cfg)
    return decide([view], getattr(scorer, "name", ""), req.return_crops, t0, cfg)


def judge_multi(req: MultiJudgeRequest, scorer: Scorer, cfg: Settings = default_settings) -> Verdict:
    """The same strike seen by several cameras: analyse each, fuse by view quality, decide once."""
    t0 = time.perf_counter()
    views = []
    for i, r in enumerate(req.views):
        if r.camera == "cam1" and i:  # keep labels unique if the client did not set them
            r = r.model_copy(update={"camera": f"cam{i + 1}"})
        views.append(analyze_view(r, scorer, cfg))
    return decide(views, getattr(scorer, "name", ""), req.return_crops, t0, cfg)
