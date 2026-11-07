"""Pose-geometry evidence for a candidate strike (punch, kick or knee).

Deterministic, on the 2D skeletons the browser tracked over the clip. Distances are in the *opponent's torso
lengths* so they are scale-free. Each criterion returns a probability (or distribution) that judge.py fuses with
CLIP's zero-shot read of the image crops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NOSE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR = 0, 7, 8, 11, 12, 13, 14, 15, 16
LHIP, RHIP, LKN, RKN, LAN, RAN = 23, 24, 25, 26, 27, 28

LOCATIONS = ["head", "torso", "below_belt", "back", "arms_or_gloves", "legs", "knee_joint", "none"]

# (tip, anchor, mid) keypoints of the striking limb
LIMB = {
    "punch": {"left": (LWR, LSH, LEL), "right": (RWR, RSH, REL)},
    "kick": {"left": (LAN, LHIP, LKN), "right": (RAN, RHIP, RKN)},
    "knee": {"left": (LKN, LHIP, None), "right": (RKN, RHIP, None)},
}
SURFACES = {
    "punch": ["knuckle_part", "open_glove", "forearm_or_elbow"],
    "kick": ["foot_or_shin", "other"],
    "knee": ["knee", "other"],
}
LEGAL_SURFACE = {"punch": "knuckle_part", "kick": "foot_or_shin", "knee": "knee"}


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def normalise(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: v / s for k, v in d.items()}


def seg_dist(p, a, b) -> float:
    ab = b - a
    n = float(ab @ ab)
    if n < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / n))
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_closest(p, a, b) -> np.ndarray:
    ab = b - a
    n = float(ab @ ab)
    if n < 1e-12:
        return a
    t = max(0.0, min(1.0, float((p - a) @ ab) / n))
    return a + t * ab


def rect_dist(p, x0, x1, y0, y1) -> float:
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dy = max(y0 - p[1], 0.0, p[1] - y1)
    return math.hypot(dx, dy)


@dataclass
class Person:
    pts: dict[int, np.ndarray]
    vis: dict[int, float]
    sh: np.ndarray = field(init=False)
    hip: np.ndarray = field(init=False)
    torso: float = field(init=False)

    def __post_init__(self):
        self.sh = (self.pts[LSH] + self.pts[RSH]) / 2
        self.hip = (self.pts[LHIP] + self.pts[RHIP]) / 2
        self.torso = max(0.02, float(np.linalg.norm(self.sh - self.hip)))

    @property
    def head_c(self) -> np.ndarray:
        return self.pts[NOSE]

    @property
    def head_r(self) -> float:
        r = 0.45 * self.torso
        if LEAR in self.pts and REAR in self.pts:
            r = max(r, 0.7 * float(np.linalg.norm(self.pts[LEAR] - self.pts[REAR])))
        return r

    @property
    def knee_y(self) -> float:
        ys = [self.pts[i][1] for i in (LKN, RKN) if i in self.pts]
        return float(np.mean(ys)) if ys else float(self.hip[1] + 0.9 * self.torso)

    def facing(self) -> int:
        """+1 faces image-right, -1 image-left, 0 unknown (nose offset from the shoulders)."""
        dx = float(self.pts[NOSE][0] - self.sh[0])
        return 0 if abs(dx) < 0.06 * self.torso else (1 if dx > 0 else -1)

    def leg_segments(self):
        for h, k, a in ((LHIP, LKN, LAN), (RHIP, RKN, RAN)):
            if h in self.pts and k in self.pts:
                yield self.pts[h], self.pts[k]
                if a in self.pts:
                    yield self.pts[k], self.pts[a]


def build_person(d: Optional[dict[str, list[float]]], aspect: float) -> Optional[Person]:
    if not d:
        return None
    pts = {int(k): np.array([v[0] * aspect, v[1]], dtype=float) for k, v in d.items()}
    vis = {int(k): float(v[2]) if len(v) > 2 else 1.0 for k, v in d.items()}
    if not all(k in pts for k in (NOSE, LSH, RSH, LHIP, RHIP)):
        return None
    return Person(pts, vis)


@dataclass
class GeoResult:
    available: bool                      # an opponent was visible at the strike peak
    thrower_p: float
    target: Optional[str]
    contact_p: float
    location_p: dict[str, float]
    glove_p: dict[str, float]            # striking-surface distribution (labels depend on the strike type)
    genuine_p: dict[str, float]
    impact_norm: tuple[float, float]     # impact point, normalised image coords
    peak_index: int
    strike_p: Optional[float] = None     # kick/knee only: probability it is a real strike, not a step
    metrics: dict = field(default_factory=dict)
    quality: float = 0.05                # how much to trust this camera view when several are fused (0..1)


def _angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / n))))


def zone_distances(w: np.ndarray, op: Person) -> dict[str, float]:
    """Distance (opponent torso lengths) from point w to each zone of the opponent."""
    t = op.torso
    xs = [op.pts[i][0] for i in (LSH, RSH, LHIP, RHIP)]
    m = 0.3 * t  # profile views have almost no shoulder width: pad for body thickness
    x0, x1 = min(xs) - m, max(xs) + m
    sh_y, hip_y = float(op.sh[1]), float(op.hip[1])
    d = {
        "head": max(0.0, float(np.linalg.norm(w - op.head_c)) - op.head_r) / t,
        "torso": rect_dist(w, x0, x1, sh_y - 0.1 * t, hip_y + 0.05 * t) / t,
        "below_belt": rect_dist(w, x0, x1, hip_y + 0.05 * t, hip_y + 0.35 * t) / t,  # groin / hip region
    }
    arm = 9.0
    for sh_i, el_i, wr_i in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        if sh_i in op.pts and el_i in op.pts:
            arm = min(arm, seg_dist(w, op.pts[sh_i], op.pts[el_i]))
        if el_i in op.pts and wr_i in op.pts:
            arm = min(arm, seg_dist(w, op.pts[el_i], op.pts[wr_i]))
    # arms always overlap the torso in profile, so they must be clearly nearest to claim the hit (+0.05 handicap)
    d["arms_or_gloves"] = max(0.0, arm - 0.12 * t) / t + 0.05
    # legs (thigh + shin) and the knee joint, which wins where the two overlap
    leg = min((seg_dist(w, a, b) for a, b in op.leg_segments()), default=9.0)
    d["legs"] = max(0.0, leg - 0.14 * t) / t + 0.03
    knee = min((float(np.linalg.norm(w - op.pts[i])) for i in (LKN, RKN) if i in op.pts), default=9.0)
    d["knee_joint"] = max(0.0, knee - 0.08 * t) / t
    return d


def _strike_points(th: Person, strike: str, side: str) -> Optional[tuple[list[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    tip_i, anc_i, mid_i = LIMB[strike][side]
    if tip_i not in th.pts or anc_i not in th.pts:
        return None
    tip, anc = th.pts[tip_i], th.pts[anc_i]
    mid = th.pts.get(mid_i) if mid_i is not None else None
    if strike == "kick" and mid is not None:
        pts = [mid + (tip - mid) * u for u in (0.0, 0.25, 0.5, 0.75, 1.0)]  # the shin sweeps the target, not just the ankle
    else:
        pts = [tip]
    return pts, tip, anc, mid


def analyze(
    times: list[float],
    people: list[dict[str, Optional[Person]]],
    aspect: float,
    thrower: str,
    side: str,
    peak_t: float,
    speed_mps: float,
    strike: str = "punch",
) -> GeoResult:
    n = len(times)
    p = int(np.argmin([abs(t - peak_t) for t in times]))
    target_id = "b" if thrower == "a" else "a"
    th = people[p].get(thrower)
    op = people[p].get(target_id)
    surf = SURFACES[strike]
    default_glove = normalise({k: (0.8 if i == 0 else 0.2 / (len(surf) - 1)) for i, k in enumerate(surf)})
    no_loc = {k: 0.0 for k in LOCATIONS}
    no_loc["none"] = 1.0
    miss = {"clean": 0, "block": 0, "miss": 1, "push": 0, "glancing": 0}

    sp = _strike_points(th, strike, side) if th is not None else None
    if sp is None:
        return GeoResult(False, 0.5, None, 0.0, no_loc, default_glove, miss, (0.5, 0.5), p, None, {"reason": "striking limb not tracked"})
    pts, tip, anc, mid = sp
    reach_dir = tip - anc
    nrm = float(np.linalg.norm(reach_dir))
    udir = reach_dir / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
    impact_norm = (float(tip[0] / aspect), float(tip[1]))
    speed_term = sig((speed_mps - 1.2) / 0.6)

    # kick / knee: is it a real strike, or just a step / lifted leg?  (CLIP gets a vote in judge.py)
    strike_p = None
    if strike == "kick":
        lift = float((th.hip[1] + 1.76 * th.torso - tip[1]) / th.torso)  # 0 = foot on the floor, >0.45 = clearly raised
        strike_p = sig((speed_mps - 1.8) / 0.5) * sig((lift - 0.45) / 0.2)
    elif strike == "knee":
        lift = float((th.hip[1] + 0.9 * th.torso - tip[1]) / th.torso)
        strike_p = sig((speed_mps - 1.8) / 0.5) * sig((lift - 0.35) / 0.15)

    if op is None:
        return GeoResult(False, 0.5 + 0.4 * speed_term, None, 0.0, no_loc, default_glove, miss, impact_norm, p, strike_p,
                         {"reason": "no opponent visible"})

    t_o = op.torso
    toward = (float(np.linalg.norm(anc - op.hip)) - float(np.linalg.norm(tip - op.hip))) / th.torso
    thrower_p = min(0.99, 0.5 * sig((toward - 0.05) / 0.12) + 0.5 * speed_term + 0.02)

    # ---- distances from the striking surface to every zone of the opponent, best of frames p-1..p+1
    best = None
    for q in [p] + [x for x in (p - 1, p + 1) if 0 <= x < n]:  # peak frame first; neighbours must be strictly closer
        thq, opq = people[q].get(thrower), people[q].get(target_id)
        spq = _strike_points(thq, strike, side) if thq is not None else None
        if spq is None or opq is None:
            continue
        # samples run knee -> ankle. The shin can pass over several zones in 2D, so the impact point is the tip-most
        # sample that is actually touching something; if nothing touches, the closest sample.
        scored_pts = []
        for pt in spq[0]:
            d = zone_distances(pt, opq)
            scored_pts.append((min(d.values()), d, pt))
        touching = [s for s in scored_pts if s[0] <= 0.05]
        m, d, pt = touching[-1] if touching else min(scored_pts, key=lambda s: s[0])
        if best is None or m < best[4] - 1e-9:
            best = (q, d, pt, opq, m)
    q_best, dists, hit_pt, op_hit, _ = best
    d_any = min(dists.values())
    contact_p = sig((0.12 - d_any) / 0.07)

    # ---- where: softmin over zones, 'none' wins when every zone is far
    logits = {k: -v / 0.08 for k, v in dists.items()}
    logits["none"] = -0.12 / 0.08
    mx = max(logits.values())
    loc = normalise({k: math.exp(v - mx) for k, v in logits.items()})
    loc["back"] = 0.0
    side_sign = 1 if th.hip[0] > op.hip[0] else -1
    f = op_hit.facing()
    back_factor = 0.9 if (f != 0 and f != side_sign) else 0.0
    moved = (loc["head"] + loc["torso"]) * back_factor
    loc["head"] *= 1 - back_factor
    loc["torso"] *= 1 - back_factor
    loc["back"] = moved
    loc = normalise(loc)

    # ---- striking surface
    if strike == "punch":
        e = mid if mid is not None else (tip + anc) / 2
        elbow_angle = _angle(anc, e, tip)
        d_elbow_any = min(zone_distances(e, op_hit).values())
        p_elbow = sig((125.0 - elbow_angle) / 12.0) * sig((d_any - d_elbow_any) / 0.06)  # folded arm, elbow nearest the target
        glove = normalise({"knuckle_part": max(0.02, 1 - p_elbow - 0.12), "open_glove": 0.12, "forearm_or_elbow": max(0.01, p_elbow)})
        extra = {"elbow_angle_deg": round(elbow_angle, 1)}
    else:
        glove = normalise({surf[0]: 0.9, surf[1]: 0.1})  # pose cannot see the surface; CLIP supplies the evidence
        extra = {}

    # ---- genuine strike components
    guards = [op_hit.pts[i] for i in (LWR, RWR) if i in op_hit.pts]  # gloves; arms/elbows count via the arms zone
    guard = min((float(np.linalg.norm(g - hit_pt)) for g in guards), default=9.0) / t_o
    in_front = any(np.linalg.norm(g - th.hip) < np.linalg.norm(op_hit.head_c - th.hip) for g in guards)
    block_sc = sig((0.22 - guard) / 0.06) * (1.0 if in_front else 0.4)
    block_p = max(block_sc, loc["arms_or_gloves"]) * contact_p

    if strike == "punch":
        other_i = RWR if side == "left" else LWR
        other_hand = float(np.linalg.norm(th.pts[other_i] - op.hip)) / t_o if other_i in th.pts else 9.0
        two_hand = sig((0.45 - other_hand) / 0.1)
    else:
        two_hand = 0.0
    push_p = contact_p * (0.7 * sig((1.3 - speed_mps) / 0.35) + 0.3 * two_hand)

    # glancing: how far the strike line passes from the centre of the zone it hit
    zone_best = min(dists, key=dists.get)
    if zone_best == "head":
        centre, radius = op_hit.head_c, op_hit.head_r
    elif zone_best in ("legs", "knee_joint"):
        cands = [seg_closest(hit_pt, a, b) for a, b in op_hit.leg_segments()]
        centre = min(cands, key=lambda c: np.linalg.norm(c - hit_pt)) if cands else (op_hit.sh + op_hit.hip) / 2
        radius = (0.14 if zone_best == "legs" else 0.1) * t_o
    elif zone_best == "below_belt":
        centre, radius = op_hit.hip + np.array([0.0, 0.2 * t_o]), 0.25 * t_o
    elif zone_best == "arms_or_gloves":
        centre, radius = hit_pt, 1.0  # the arm was struck wherever it is: not a glancing call
    else:
        centre, radius = (op_hit.sh + op_hit.hip) / 2, 0.5 * t_o
    v = centre - anc
    perp = abs(float(udir[0] * v[1] - udir[1] * v[0]))
    glance_p = contact_p * sig((perp / max(radius, 1e-3) - 0.55) / 0.12)

    # reaction: opponent's head moves along the strike direction after contact
    recoil_term = 0.5
    after = [q for q in range(n) if peak_t < times[q] <= peak_t + 300 and people[q].get(target_id)]
    before = [q for q in range(n) if peak_t - 300 <= times[q] <= peak_t and people[q].get(target_id)]
    if after and before:
        h0 = people[before[0]][target_id].head_c
        disp = max(float((people[q][target_id].head_c - h0) @ udir) for q in after) / t_o
        recoil_term = sig((disp - 0.06) / 0.05)

    clean = contact_p * (1 - block_p) * (1 - push_p) * (1 - glance_p) * (0.55 + 0.45 * recoil_term)
    genuine = normalise({
        "clean": max(clean, 1e-4),
        "block": max(block_p * (1 - push_p), 1e-4),
        "miss": max(1 - contact_p, 1e-4),
        "push": max(push_p, 1e-4),
        "glancing": max(glance_p * (1 - block_p), 1e-4),
    })

    # ---- view quality, used to weight this camera against others (multi-camera fusion)
    #  * visibility of the joints that decide the call (striking limb, opponent's head and trunk)
    #  * foreshortening: a limb pointing at the lens looks short in 2D and hides the gap between glove and face, so
    #    contact cannot be judged; a camera seeing the strike side-on is the one to believe.
    key_vis = [th.vis.get(i, 0.0) for i in (tip_i_of(strike, side), anc_i_of(strike, side))]
    key_vis += [op.vis.get(i, 0.0) for i in (NOSE, LSH, LHIP)]
    vq = float(np.mean(key_vis))
    expected = {"punch": 1.15, "kick": 1.76}.get(strike)  # full-extension limb length in torso lengths
    fs = sig((nrm / (expected * th.torso) - 0.45) / 0.12) if expected else 1.0
    quality = float(np.clip(vq * fs, 0.05, 1.0))

    return GeoResult(
        True, thrower_p, target_id, contact_p, loc, glove, genuine, impact_norm, p, strike_p,
        {
            "nearest_zone_dist_torso": round(d_any, 3),
            "zone_dists": {k: round(v, 3) for k, v in dists.items()},
            "guard_dist_torso": round(guard, 3),
            "perp_ratio": round(perp / max(radius, 1e-3), 3),
            "recoil_term": round(recoil_term, 3),
            "speed_mps": round(speed_mps, 2),
            "opponent_facing": f,
            "back_factor": back_factor,
            "visibility": round(vq, 3),
            "foreshortening": round(fs, 3),
            **extra,
        },
        quality=quality,
    )


def tip_i_of(strike: str, side: str) -> int:
    return LIMB[strike][side][0]


def anc_i_of(strike: str, side: str) -> int:
    return LIMB[strike][side][1]
