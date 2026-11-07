import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

torch = pytest.importorskip("torch")

from app import throw_clip as tc  # noqa: E402
from app.clip_model import UniformScorer  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeScorer(UniformScorer):
    """embed() returns deterministic unit vectors instead of running CLIP"""

    def __init__(self):
        self.calls = []

    def embed(self, images):
        self.calls.append(len(images))
        g = torch.Generator().manual_seed(len(images))
        f = torch.randn(len(images), 512, generator=g)
        return f / f.norm(dim=1, keepdim=True)


def jpeg(w=320, h=180):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def model(tmp_path):
    classes = ["ippon_seoi_nage", "o_goshi", "osoto_gari", "uchi_mata"]
    net = tc.build_head(4)
    p = tmp_path / "m.pt"
    torch.save({"model_state_dict": net.state_dict(), "classes": classes, "dim": 512, "hidden": 128, "best_epoch": 3, "val": {"accuracy": 0.5}}, p)
    return tc.ThrowTypeModel.load(str(p))


def test_letterbox_pads_to_a_square_without_cropping():
    wide = Image.new("RGB", (160, 90), (255, 0, 0))
    sq = tc.letterbox(wide)
    assert sq.size == (160, 160)
    assert sq.getpixel((80, 80)) == (255, 0, 0)          # the picture is intact in the middle
    assert sq.getpixel((80, 2)) == (114, 114, 114)       # grey bars above and below, nothing cut off
    assert tc.letterbox(Image.new("RGB", (50, 50))).size == (50, 50)


def test_frame_picking_is_evenly_spaced_and_handles_short_clips():
    assert tc.pick_frames(100) == [0, 14, 28, 42, 57, 71, 85, 99]
    assert tc.pick_frames(8) == list(range(8))
    assert tc.pick_frames(3) == [0, 0, 1, 1, 1, 1, 2, 2] and len(tc.pick_frames(1)) == 8


def test_model_predicts_probabilities_with_pretty_names(tmp_path):
    m, s = model(tmp_path), FakeScorer()
    images = [Image.new("RGB", (320, 180), (i * 10, 50, 50)) for i in range(13)]
    r = m.predict(images, s)
    assert s.calls == [8]  # exactly 8 frames are embedded, however many were sent
    assert set(r["probs"]) == {"Ippon-seoi-nage", "O-goshi", "Osoto-gari", "Uchi-mata"} and abs(sum(r["probs"].values()) - 1) < 1e-5
    assert r["technique"] in r["probs"] and 0 < r["confidence"] <= 1 and r["key"] in m.classes


def test_endpoint_returns_a_technique_and_is_503_without_a_model(tmp_path):
    frames = [{"t_ms": i * 100, "jpeg_b64": jpeg()} for i in range(10)]
    with TestClient(create_app(FakeScorer(), throw_model=model(tmp_path))) as c:
        assert c.get("/api/health").json()["throw_model"] is True
        body = c.post("/api/throw_type", json={"frames": frames}).json()
        assert body["technique"] in {"Ippon-seoi-nage", "O-goshi", "Osoto-gari", "Uchi-mata"} and body["model_info"]["best_epoch"] == 3
        assert c.post("/api/throw_type", json={"frames": frames[:2]}).status_code == 422
    with TestClient(create_app(FakeScorer())) as c:
        assert c.get("/api/health").json()["throw_model"] is False
        assert c.post("/api/throw_type", json={"frames": frames}).status_code == 503
