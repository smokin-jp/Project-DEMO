from fastapi.testclient import TestClient

from app.clip_model import UniformScorer
from app.judge import fuse, judge
from app.main import create_app
from app.schemas import JudgeRequest

from .clips import make_request


def run(**kw):
    return judge(JudgeRequest(**make_request(**kw)), UniformScorer())


def test_clean_head_punch_scores():
    v = run(scenario="head", speed=4.5, recoil=0.05)
    assert v.thrower.value == "a" and v.intended_target.value == "b"
    assert v.contact.value is True
    assert v.contact_location.value == "head"
    assert v.legal_area.value is True
    assert v.legal_glove_part.value is True
    assert v.genuine_punch.value is True and v.genuine_punch.detail["classification"] == "clean"
    assert v.scored and not v.review and v.foul is None
    assert set(v.crops) >= {"frame", "impact", "weapon", "reaction"}


def test_body_punch_scores():
    v = run(scenario="torso", speed=4.5, recoil=0.05)
    assert v.contact.value and v.contact_location.value == "torso" and v.legal_area.value
    assert v.scored


def test_miss_is_not_contact():
    v = run(scenario="miss")
    assert v.contact.value is False
    assert v.genuine_punch.detail["classification"] == "miss"
    assert not v.scored and v.foul is None


def test_low_blow_is_illegal_area_and_a_foul():
    v = run(scenario="low", speed=4.5, recoil=0.05)
    assert v.contact.value and v.contact_location.value == "below_belt"
    assert v.legal_area.value is False
    assert not v.scored
    assert v.foul == "low_blow"


def test_blocked_punch_does_not_score():
    v = run(scenario="block", speed=4.5)
    assert v.contact.value
    assert v.genuine_punch.detail["classification"] == "block"
    assert not v.scored and v.foul is None


def test_slow_two_handed_push_is_not_a_punch():
    v = run(scenario="push", speed=0.5)
    assert v.genuine_punch.detail["classification"] == "push"
    assert not v.scored


def test_glancing_blow_does_not_score():
    v = run(scenario="glance", speed=4.5)
    assert v.genuine_punch.detail["classification"] in ("glancing", "miss")
    assert not v.scored


def test_no_opponent_nothing_to_score():
    v = run(scenario="head", opponent=False)
    assert v.intended_target.value is None
    assert v.contact.value is False and not v.scored


def test_blue_can_be_the_thrower():
    v = judge(JudgeRequest(**make_request(scenario="head", thrower="b")), UniformScorer())
    assert v.thrower.value == "b" and v.intended_target.value == "a"


def test_fusion_is_a_soft_and():
    geo = {"x": 0.9, "y": 0.1}
    agree = fuse(geo, {"x": 0.8, "y": 0.2}, 0.6)
    clash = fuse(geo, {"x": 0.1, "y": 0.9}, 0.6)
    assert agree["x"] > 0.85 and agree["x"] > clash["x"] > 0.5  # geometry still leads; CLIP disagreement pulls it down
    assert fuse(geo, None, 0.6) == geo or abs(fuse(geo, None, 0.6)["x"] - 0.9) < 1e-9


def test_api_roundtrip_and_validation():
    client = TestClient(create_app(UniformScorer()))
    with client:
        assert client.get("/api/health").json()["ok"]
        r = client.post("/api/judge", json=make_request(scenario="head", recoil=0.05))
        assert r.status_code == 200
        body = r.json()
        for k in ("thrower", "intended_target", "contact", "contact_location", "legal_area", "legal_glove_part", "genuine_punch"):
            assert "value" in body[k] and "confidence" in body[k]
        assert body["scored"] is True
        bad = make_request()
        bad["frames"] = bad["frames"][:2]
        assert client.post("/api/judge", json=bad).status_code == 422


# ------------------------------------------------------------------ kicks and knees
def run_k(scenario, strike="kick", speed=5.0, legal=None, **kw):
    return judge(JudgeRequest(**make_request(scenario=scenario, strike=strike, speed=speed, legal_zones=legal, **kw)), UniformScorer())


def test_high_kick_to_head_scores():
    v = run_k("head", recoil=0.05)
    assert v.strike == "kick"
    assert v.strike_check.value is True
    assert v.contact.value and v.contact_location.value == "head" and v.legal_area.value
    assert v.legal_glove_part.value is True and v.legal_glove_part.detail["part"] == "foot_or_shin"
    assert v.scored and v.foul is None


def test_body_kick_scores():
    v = run_k("torso", recoil=0.05)
    assert v.contact_location.value == "torso" and v.scored


def test_leg_kick_is_a_foul_where_legs_are_not_legal_and_a_score_where_they_are():
    full_contact = run_k("thigh", legal=["head", "torso"], recoil=0.05)
    assert full_contact.contact_location.value == "legs" and full_contact.legal_area.value is False
    assert not full_contact.scored and full_contact.foul == "kick_below_waist"
    low_kick = run_k("thigh", legal=["head", "torso", "legs"], recoil=0.05)
    assert low_kick.legal_area.value is True and low_kick.scored and low_kick.foul is None


def test_thrusting_kick_to_knee_joint_is_a_foul():
    v = run_k("knee", legal=["head", "torso", "legs"], recoil=0.05)
    assert v.contact_location.value == "knee_joint"
    assert not v.scored and v.foul == "knee_joint_kick"


def test_groin_kick_is_a_low_blow():
    v = run_k("groin", legal=["head", "torso", "legs"], recoil=0.05)
    assert v.contact_location.value == "below_belt" and v.foul == "low_blow" and not v.scored


def test_kick_that_misses_is_no_contact():
    v = run_k("kmiss")
    assert v.strike_check.value is True and v.contact.value is False and not v.scored and v.foul is None


def test_a_step_is_not_a_kick():
    v = run_k("step", speed=0.9)
    assert v.strike_check.value is False
    assert not v.scored and v.foul is None and not v.review
    assert "not a real kick" in v.summary


def test_knee_strike_to_the_body_scores():
    v = run_k("torso", strike="knee", legal=["head", "torso", "legs"], recoil=0.05)
    assert v.strike == "knee" and v.strike_check.value is True
    assert v.contact.value and v.contact_location.value == "torso"
    assert v.legal_glove_part.detail["part"] == "knee" and v.scored


def test_kick_judged_with_blue_as_thrower_and_api_carries_new_fields():
    client = TestClient(create_app(UniformScorer()))
    with client:
        req = make_request(scenario="head", strike="kick", speed=5.0, thrower="b", recoil=0.05)
        body = client.post("/api/judge", json=req).json()
        assert body["strike"] == "kick" and body["strike_check"]["value"] is True
        assert body["thrower"]["value"] == "b" and body["intended_target"]["value"] == "a"


def test_startup_path_with_model_loading_and_warmup(monkeypatch):
    """The lifespan (load + warm-up) only runs for the real scorer; exercise it with a fake model so a broken import
    in that path cannot hide behind the stub scorer the other tests inject."""
    import app.main as main

    class FakeClip(UniformScorer):
        loaded = True

        def __init__(self, *a, **k):
            pass

        def load(self):
            return self

    monkeypatch.setattr(main, "ClipScorer", FakeClip)
    with TestClient(main.create_app()) as c:
        assert c.get("/api/health").json()["loaded"] is True


# ------------------------------------------------------------------ several cameras
from app.judge import judge_multi
from app.schemas import MultiJudgeRequest


def multi(*views, **kw):
    reqs = [make_request(camera=f"cam{i + 1}", **v) for i, v in enumerate(views)]
    return judge_multi(MultiJudgeRequest(views=reqs, **kw), UniformScorer())


def test_one_camera_through_the_multi_path_equals_the_single_path():
    single = run(scenario="head", speed=4.5, recoil=0.05)
    m = multi(dict(scenario="head", speed=4.5, recoil=0.05))
    assert (m.scored, m.confidence, m.contact_location.value) == (single.scored, single.confidence, single.contact_location.value)
    assert m.views == [] and "frame" in m.crops  # no per-camera prefixes or summary for a single view


def test_two_agreeing_cameras_score_and_report_each_view():
    v = multi(dict(scenario="head", speed=4.5, recoil=0.05), dict(scenario="head", speed=4.5, recoil=0.05))
    assert v.scored and v.contact.value and v.contact_location.value == "head"
    assert [x["camera"] for x in v.views] == ["cam1", "cam2"] and all(x["available"] for x in v.views)
    assert {"cam1_frame", "cam1_impact", "cam2_frame", "cam2_weapon"} <= set(v.crops)
    assert "[2 cameras]" in v.summary


def test_a_clear_miss_from_a_good_camera_vetoes_an_apparent_hit_from_a_poor_one():
    # camera 1 sees an "overlap" through poorly tracked joints (low visibility); camera 2 sees the gap clearly
    # (a full-length punch that stops 0.12 short of an opponent who has stepped back: a clean, well-seen miss)
    v = multi(dict(scenario="head", speed=4.5, recoil=0.05, vis=0.3), dict(scenario="head", speed=4.5, opp_dx=0.12))
    assert v.views[0]["quality"] < 0.5 <= v.views[1]["quality"]
    assert v.contact.value is False and not v.scored
    assert v.contact.detail["veto_by_good_view"] is True
    assert v.genuine_punch.detail["classification"] == "miss"


def test_a_good_camera_hit_is_not_cancelled_by_a_poor_miss():
    v = multi(dict(scenario="head", speed=4.5, recoil=0.05), dict(scenario="head", speed=4.5, opp_dx=0.12, vis=0.2))
    assert v.views[1]["quality"] < v.views[0]["quality"]
    assert v.contact.value is True and v.scored


def test_a_limb_pointing_at_the_lens_is_downweighted():
    v = multi(dict(scenario="head", speed=4.5, recoil=0.05), dict(scenario="pointing", speed=4.5))
    assert v.views[1]["foreshortening"] < 0.5 and v.views[1]["quality"] < 0.5 < v.views[0]["quality"]
    assert v.scored  # the side-on camera decides


def test_a_camera_that_cannot_see_the_opponent_is_ignored():
    v = multi(dict(scenario="head", speed=4.5, recoil=0.05), dict(scenario="head", speed=4.5, opponent=False))
    assert v.scored
    assert [x["available"] for x in v.views] == [True, False]


def test_no_camera_sees_both_fighters():
    v = multi(dict(scenario="head", opponent=False), dict(scenario="head", opponent=False))
    assert v.intended_target.value is None and not v.scored and v.contact.value is False


def test_kicks_fuse_across_cameras_too():
    kick = dict(scenario="head", strike="kick", speed=5.0, recoil=0.05)
    v = multi(kick, kick)
    assert v.strike == "kick" and v.strike_check.value is True and v.scored
    step = dict(scenario="step", strike="kick", speed=0.9)
    assert multi(step, step).strike_check.value is False


def test_multi_api_roundtrip_and_validation():
    client = TestClient(create_app(UniformScorer()))
    with client:
        views = [make_request(scenario="head", speed=4.5, recoil=0.05, camera="left"), make_request(scenario="head", speed=4.5, recoil=0.05, camera="right")]
        r = client.post("/api/judge_multi", json={"views": views})
        assert r.status_code == 200
        body = r.json()
        assert body["scored"] is True and [x["camera"] for x in body["views"]] == ["left", "right"]
        assert "left_frame" in body["crops"] and "right_impact" in body["crops"]
        assert client.post("/api/judge_multi", json={"views": []}).status_code == 422
        assert client.post("/api/judge_multi", json={"views": views * 3}).status_code == 422  # max 4 cameras
