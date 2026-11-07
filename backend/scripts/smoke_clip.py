"""Loads the real CLIP checkpoint and judges a synthetic clip: checks the model works and measures latency.
   cd backend && .venv/Scripts/python -m scripts.smoke_clip
"""
import time

from app.clip_model import ClipScorer
from app.config import settings
from app.judge import judge
from app.schemas import JudgeRequest
from tests.clips import make_request

t = time.perf_counter()
scorer = ClipScorer(settings.clip_model, settings.device).load()
print(f"loaded {settings.clip_model} on {settings.device} in {time.perf_counter() - t:.1f}s", flush=True)

req = JudgeRequest(**make_request(scenario="head", speed=4.5, recoil=0.05))
judge(req, scorer)  # warm-up
for sc in ("head", "low", "miss"):
    req = JudgeRequest(**make_request(scenario=sc, speed=4.5))
    t = time.perf_counter()
    v = judge(req, scorer)
    print(f"{sc:5s} {1000 * (time.perf_counter() - t):6.0f} ms  scored={v.scored} review={v.review} conf={v.confidence} foul={v.foul}")
    print("      contact clip:", v.contact.detail["clip"], "| genuine clip:", v.genuine_punch.detail["clip"], flush=True)
