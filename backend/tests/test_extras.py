"""The team's trained models inside the judge: kick detector (CLIP B/32 + logistic regression) and punch-type GRU."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")

from app import extras as ex_mod  # noqa: E402
from app.clip_model import UniformScorer  # noqa: E402
from app.extras import Extras, fighters_crop, load_extras  # noqa: E402
from app.judge import judge, judge_multi  # noqa: E402
from app.kick_clf import KickClassifier  # noqa: E402
from app.main import create_app  # noqa: E402
from app.punch_gru import CHECKPOINT, CLASSES, MIN_VALID_FRAMES, PunchTypeModel, coco_keypoints, resample  # noqa: E402
from app.schemas import JudgeRequest, MultiJudgeRequest  # noqa: E402
from PIL import Image  # noqa: E402

from .clips import make_request  # noqa: E402


class FakeEmbedder:
    """deterministic 512-d unit vectors instead of running CLIP B/32"""

    def images(self, images):
        out = np.stack([np.random.default_rng(len(i.tobytes()) % 997 + k).normal(size=512) for k, i in enumerate(images)]) if images else np.zeros((0, 512))
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32) if len(out) else out

    def texts(self, prompts):
        out = np.stack([np.random.default_rng(len(p)).normal(size=512) for p in prompts])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


class Scripted:
    """a classifier whose predict_proba follows a script, one probability per picture"""

    def __init__(self, probs):
        self.probs, self.i = list(probs), 0

    def predict_proba(self, emb):
        n = len(emb)
        p = np.array([self.probs[(self.i + k) % len(self.probs)] for k in range(n)])
        self.i += n
        return np.stack([1 - p, p], axis=1)


def kick_model(probs, threshold=0.5):
    return KickClassifier(Scripted(probs), threshold, FakeEmbedder())


IMGS = [Image.new("RGB", (64 + k, 64), (k * 20, 80, 120)) for k in range(5)]


# --------------------------------------------------------------------------------------------- kick detector
def test_rolling_window_needs_three_positive_frames_of_five():
    assert kick_model([0.9, 0.9, 0.9, 0.1, 0.1]).judge_frames(IMGS)["detected"] is True
    r = kick_model([0.9, 0.9, 0.1, 0.1, 0.1]).judge_frames(IMGS)
    assert r["detected"] is False and r["positive_frames"] == 2 and r["frames_considered"] == 5
    assert kick_model([0.2]).judge_frames([])["mean"] is None


def test_real_joblib_model_loads_and_scores_with_a_stand_in_embedder():
    m = KickClassifier.load(FakeEmbedder())
    s = m.scores(IMGS)
    assert len(s) == 5 and all(0.0 <= x <= 1.0 for x in s)
    assert 0.4 < m.threshold < 0.6  # the tuned threshold shipped with the model


def test_trained_kick_vote_moves_the_is_it_a_real_kick_check_and_is_reported():
    req = JudgeRequest(**make_request(scenario="torso", strike="kick", speed=5.0))
    plain = judge(req, UniformScorer())
    yes = judge(req, UniformScorer(), extras=Extras(kick=kick_model([0.97])))
    no = judge(req, UniformScorer(), extras=Extras(kick=kick_model([0.03])))
    py, pp, pn = (x.strike_check.detail["p_real"] for x in (yes, plain, no))
    assert pn < pp - 0.2 and py > pn + 0.2  # a confident "no kick" pulls the check down hard; a "kick" vote agrees with the geometry (which was already sure)
    assert py == pytest.approx(pp, abs=0.02)
    tk = yes.strike_check.detail["trained_kick"]
    assert tk["detected"] is True and tk["frames_considered"] <= 5 and tk["mean"] == pytest.approx(0.97, abs=0.01)
    assert "trained_kick" not in plain.strike_check.detail


def test_a_strong_trained_no_kick_can_turn_a_borderline_swing_into_not_a_kick():
    req = JudgeRequest(**make_request(scenario="torso", strike="kick", speed=3.0))  # slowish swing: geometry unsure
    assert judge(req, UniformScorer(), extras=Extras(kick=kick_model([0.02]))).strike_check.value is False


def test_kick_detector_is_not_used_for_punches_or_knees():
    kick = Extras(kick=kick_model([0.9]))
    assert judge(JudgeRequest(**make_request(scenario="head", strike="punch")), UniformScorer(), extras=kick).punch_type is None
    knee = judge(JudgeRequest(**make_request(scenario="torso", strike="knee", speed=5.0)), UniformScorer(), extras=kick)
    assert "trained_kick" not in knee.strike_check.detail


def test_fighters_crop_centres_on_the_fighters_and_falls_back_to_the_frame():
    img = Image.new("RGB", (1600, 900))
    c = fighters_crop(img, [(0.3, 0.3), (0.7, 0.8)])
    assert c.size[0] == c.size[1] and c.size[0] >= int(0.45 * 900)
    assert fighters_crop(img, []) is img


# ---------------------------------------------------------------------------------------------- punch type
@pytest.fixture(scope="module")
def punch_model():
    return PunchTypeModel.load(CHECKPOINT, embedder=None)


def pose(ext=0.0, eyes=True):
    p = {str(i): [0.5, 0.5, 1.0] for i in range(33)}
    p.update({"0": [0.5, 0.25, 1], "11": [0.45, 0.35, 1], "12": [0.55, 0.35, 1], "23": [0.46, 0.6, 1], "24": [0.54, 0.6, 1],
              "13": [0.42, 0.45, 1], "14": [0.58 + 0.1 * ext, 0.42, 1], "15": [0.44, 0.38, 1], "16": [0.6 + 0.2 * ext, 0.36, 1],
              "25": [0.46, 0.78, 1], "26": [0.54, 0.78, 1], "27": [0.46, 0.95, 1], "28": [0.54, 0.95, 1], "7": [0.46, 0.25, 1], "8": [0.54, 0.25, 1]})
    if eyes:
        p.update({"2": [0.48, 0.24, 1], "5": [0.52, 0.24, 1]})
    else:
        p.pop("2"), p.pop("5")
    return p


def test_mediapipe_to_coco_mapping_and_eye_fallback():
    a = coco_keypoints(pose(eyes=True), 2.0)
    b = coco_keypoints(pose(eyes=False), 2.0)  # an older client that does not send the eyes: the nose stands in
    assert a.shape == (17, 2) and a[5, 0] == pytest.approx(0.45 * 2.0) and a[11, 1] == pytest.approx(0.6)  # left shoulder, left hip (COCO order)
    assert np.allclose(b[1], b[0]) and np.allclose(b[2], b[0])
    assert coco_keypoints(None, 1.0) is None


def test_resample_gives_25_even_frames_and_needs_enough_valid_frames():
    times = [i * 40.0 for i in range(10)]
    frames = [coco_keypoints(pose(i / 9), 1.0) for i in range(10)]
    seq = resample(times, frames)
    assert seq.shape == (25, 17, 2)
    assert np.allclose(seq[0], frames[0]) and np.allclose(seq[-1], frames[-1])
    assert resample(times, [None] * 10) is None
    assert resample(times[: MIN_VALID_FRAMES - 1], frames[: MIN_VALID_FRAMES - 1]) is None
    gappy = [f if i % 3 else None for i, f in enumerate(frames)]  # a third of the frames missing: interpolated across
    assert resample(times, gappy).shape == (25, 17, 2)


def test_gru_features_are_hip_centred_torso_scaled_with_zero_first_velocity(punch_model):
    seq = resample([i * 33.0 for i in range(20)], [coco_keypoints(pose(np.sin(np.pi * i / 19)), 16 / 9) for i in range(20)])
    f = punch_model.features(seq)
    assert f.shape == (25, 17, 4) and np.isfinite(f).all()
    assert np.allclose(f[0, :, 2:], 0.0)
    hip = (f[:, 11, :2] + f[:, 12, :2]) / 2
    assert np.allclose(np.median(hip, axis=0), 0.0, atol=1e-5)


def test_punch_type_is_returned_with_the_verdict_and_never_changes_the_score(punch_model):
    req = JudgeRequest(**make_request(scenario="head", strike="punch", speed=4.5, recoil=0.05))
    plain = judge(req, UniformScorer())
    withp = judge(req, UniformScorer(), extras=Extras(punch=punch_model))
    pt = withp.punch_type
    assert pt["label"] in CLASSES and 0 < pt["confidence"] <= 1 and sum(pt["probs"].values()) == pytest.approx(1, abs=0.02)
    assert pt["source"] == "skeleton GRU" and "experimental" in pt["note"]
    assert (withp.scored, withp.foul, withp.confidence, withp.contact.value) == (plain.scored, plain.foul, plain.confidence, plain.contact.value)
    assert plain.punch_type is None


def test_punch_type_blends_clip_when_an_embedder_is_present(punch_model):
    m = PunchTypeModel(punch_model.net, punch_model.meta, FakeEmbedder())
    seq = resample([i * 33.0 for i in range(20)], [coco_keypoints(pose(np.sin(np.pi * i / 19)), 1.0) for i in range(20)])
    r = m.predict(seq, [Image.new("RGB", (64, 64))])
    assert r["clip"] is not None and "80/20" in r["source"]
    g = m.predict(seq, None)
    assert g["clip"] is None and g["gru"] == r["gru"]


def test_punch_type_is_skipped_when_the_thrower_is_barely_tracked(punch_model):
    d = make_request(scenario="head", strike="punch", speed=4.5)
    for f in d["frames"]:
        f["poses"]["a"] = None
    v = judge(JudgeRequest(**d), UniformScorer(), extras=Extras(punch=punch_model))
    assert v.punch_type is None


def test_multi_camera_carries_the_punch_type_and_kick_check(punch_model):
    req = MultiJudgeRequest(views=[JudgeRequest(**make_request(scenario="head", strike="punch", camera=f"cam{i}")) for i in (1, 2)])
    assert judge_multi(req, UniformScorer(), extras=Extras(punch=punch_model)).punch_type["label"] in CLASSES
    kreq = MultiJudgeRequest(views=[JudgeRequest(**make_request(scenario="torso", strike="kick", speed=5.0, camera=f"cam{i}")) for i in (1, 2)])
    assert "trained_kick" in judge_multi(kreq, UniformScorer(), extras=Extras(kick=kick_model([0.9]))).strike_check.detail


# ------------------------------------------------------------------------------------- robustness / wiring
def test_a_broken_trained_model_never_breaks_a_judgement():
    class Boom:
        def judge_frames(self, imgs):
            raise RuntimeError("model exploded")

        def predict(self, *a, **k):
            raise RuntimeError("model exploded")

    k = judge(JudgeRequest(**make_request(scenario="torso", strike="kick", speed=5.0)), UniformScorer(), extras=Extras(kick=Boom()))
    assert k.strike == "kick" and "trained_kick" not in k.strike_check.detail
    p = judge(JudgeRequest(**make_request(scenario="head")), UniformScorer(), extras=Extras(punch=Boom()))
    assert p.punch_type is None and p.contact.value is not None


def test_health_reports_which_trained_models_are_loaded(punch_model):
    with TestClient(create_app(UniformScorer(), extras=Extras(punch=punch_model))) as c:
        h = c.get("/api/health").json()
        assert h["punch_model"] is True and h["kick_model"] is False
    with TestClient(create_app(UniformScorer())) as c:  # tests never load the heavy extras on their own
        assert c.get("/api/health").json()["kick_model"] is False


def test_api_judge_returns_the_new_fields(punch_model):
    with TestClient(create_app(UniformScorer(), extras=Extras(punch=punch_model, kick=kick_model([0.9])))) as c:
        r = c.post("/api/judge", json=make_request(scenario="head", strike="punch", speed=4.5))
        assert r.status_code == 200 and r.json()["punch_type"]["label"] in CLASSES
        r = c.post("/api/judge", json=make_request(scenario="torso", strike="kick", speed=5.0))
        assert r.json()["strike_check"]["detail"]["trained_kick"]["frames_considered"] >= 1


def test_load_extras_survives_missing_clip_b32_weights(monkeypatch):
    def no_weights(self):
        raise OSError("no internet and no cached ViT-B/32")

    monkeypatch.setattr(ex_mod.B32Embedder, "load", no_weights)
    ex = load_extras()
    assert ex.kick is None and ex.punch is not None and ex.punch.embedder is None  # the punch GRU needs no CLIP
    assert ex.names == {"kick_model": False, "punch_model": True}


def test_b32_embedder_works_from_a_worker_thread_where_gradients_are_on():
    """regression: grad mode is per thread, and the API runs requests in worker threads (numpy() failed on a grad tensor)"""
    import threading

    import torch

    from app.b32_clip import B32Embedder

    class StubModel:
        lin = torch.nn.Linear(3, 512)

        def get_image_features(self, pixel_values):
            return self.lin(torch.ones(pixel_values.shape[0], 3))

    class StubProc:
        def __call__(self, images, return_tensors):
            class Out(dict):
                def to(self, d):
                    return self

            return Out(pixel_values=torch.zeros(len(images), 3, 4, 4))

    emb = B32Embedder()
    emb._model, emb._proc = StubModel(), StubProc()
    result = {}

    def worker():
        torch.set_grad_enabled(True)
        try:
            result["v"] = emb.images([Image.new("RGB", (8, 8))] * 2)
        except Exception as e:  # noqa: BLE001
            result["err"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert "err" not in result, result.get("err")
    assert result["v"].shape == (2, 512) and np.allclose(np.linalg.norm(result["v"], axis=1), 1.0, atol=1e-5)
