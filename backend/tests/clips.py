"""Synthetic strike clips (stick-figure poses + blank JPEG frames) for testing geometry/fusion/API logic.

These test the *logic*; they say nothing about CLIP accuracy on real footage.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

W, H = 640, 360
ASPECT = W / H
TORSO = 0.17  # image-height units
FPS = 30
N = 13
PEAK = 6


def _jpeg() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (W, H), (90, 90, 90)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def fighter(hip_x: float, face: int, *, wrist_l=None, wrist_r=None, head_dx=0.0, elbow_r=None, knee_r=None, ankle_r=None, vis=1.0):
    """Isotropic coords -> compact normalised pose dict. Profile view, hip at y=0.55, legs ~0.30 long."""
    hy = 0.55
    sh = (hip_x, hy - TORSO)
    nose = (hip_x + face * 0.025 + head_dx, hy - TORSO - 0.075)
    wl = wrist_l or (hip_x + face * 0.09, hy - TORSO + 0.03)
    wr = wrist_r or (hip_x + face * 0.07, hy - TORSO + 0.05)
    el = (hip_x + face * 0.03, hy - TORSO + 0.10)
    er = elbow_r or (hip_x + face * 0.04, hy - TORSO + 0.11)
    pts = {
        0: nose, 7: (nose[0] - 0.02, nose[1]), 8: (nose[0] + 0.02, nose[1]),
        11: (sh[0] - 0.01, sh[1]), 12: (sh[0] + 0.01, sh[1]),
        13: el, 14: er, 15: wl, 16: wr,
        23: (hip_x - 0.01, hy), 24: (hip_x + 0.01, hy),
        25: (hip_x + 0.02, hy + 0.15), 26: knee_r or (hip_x - 0.02, hy + 0.15),
        27: (hip_x + 0.03, hy + 0.30), 28: ankle_r or (hip_x - 0.03, hy + 0.30),
    }
    return {str(k): [v[0] / ASPECT, v[1], vis] for k, v in pts.items()}


def lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def make_request(
    scenario: str = "head",
    *,
    speed: float = 4.0,
    opponent: bool = True,
    guard: bool = False,
    recoil: float = 0.0,
    thrower: str = "a",
    strike: str = "punch",
    legal_zones: list[str] | None = None,
    vis: float = 1.0,
    camera: str = "cam1",
    opp_dx: float = 0.0,
) -> dict:
    """Red (a, left, faces right) strikes Blue (b, right, faces left) with the right hand / leg."""
    ax, bx = 0.30, 0.46          # hip x in image-height units (gap 0.16 = ~1 torso)
    hy = 0.55
    sh_y = hy - TORSO
    head = (bx - 0.025, sh_y - 0.075)
    targets = {
        # punches / generic
        "head": head,
        "torso": (bx, sh_y + 0.07),
        "low": (bx, hy + 0.07),
        "miss": (bx - 0.20, sh_y - 0.02),
        "glance": (bx - 0.025, sh_y - 0.075 - 0.075),
        # kicks / knees
        "groin": (bx, hy + 0.03),
        "thigh": (bx + 0.01, hy + 0.64 - 0.55 + 0.0),
        "knee": (bx + 0.02, hy + 0.15),
        "step": (ax + 0.10, hy + 0.27),
        "kmiss": (bx - 0.30, sh_y + 0.1),
        # limb pointing along the camera axis: it barely moves in the image, so contact cannot be judged from this view
        "pointing": (ax + 0.025, sh_y + 0.05),
    }
    key = "head" if scenario in ("block", "push") else scenario
    tgt = targets[key]
    if scenario == "block":
        guard = True

    if strike == "kick":
        rest = (ax - 0.03, hy + 0.30)
    elif strike == "knee":
        rest = (ax - 0.02, hy + 0.15)
    else:
        rest = (ax + 0.07, sh_y + 0.05)

    frames = []
    for i in range(N):
        t = i * 1000 / FPS
        if i <= PEAK:
            u = i / PEAK
            u = u * u * (3 - 2 * u)
            tip = lerp(rest, tgt, u)
        else:
            u = (i - PEAK) / (N - 1 - PEAK)
            tip = lerp(tgt, rest, u)
        head_dx = recoil * min(1.0, (i - PEAK) / 3) if recoil and i > PEAK else 0.0
        opp_wr = (tgt[0] + 0.015, tgt[1] + 0.01) if guard else None

        kw = {}
        if strike == "punch":
            kw = dict(wrist_r=tip, elbow_r=(tip[0] - 0.09, tip[1] + 0.07))
        elif strike == "kick":
            hip = (ax + 0.01, hy)
            kw = dict(ankle_r=tip, knee_r=((hip[0] + tip[0]) / 2, (hip[1] + tip[1]) / 2 - 0.02))
        else:  # knee: tip IS the knee, shin hangs below it
            kw = dict(knee_r=tip, ankle_r=(tip[0] - 0.02, tip[1] + 0.15))
        poses = {thrower: fighter(ax if thrower == "a" else bx, 1 if thrower == "a" else -1, vis=vis, **kw)}
        other = "b" if thrower == "a" else "a"
        poses[other] = fighter(bx + opp_dx, -1, wrist_l=opp_wr, wrist_r=opp_wr, head_dx=head_dx, vis=vis) if opponent else None
        frames.append({"t_ms": t, "jpeg_b64": _jpeg(), "poses": poses})
    ev = {"thrower": thrower, "hand": "right", "strike": strike, "peak_t_ms": PEAK * 1000 / FPS, "speed_mps": speed, "extension_m": 0.4}
    if legal_zones is not None:
        ev["legal_zones"] = legal_zones
    return {"frames": frames, "evidence": ev, "return_crops": True, "camera": camera}
