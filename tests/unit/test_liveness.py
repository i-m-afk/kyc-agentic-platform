import os
import pytest
from src.agents.liveness import verify_liveness
from src.schemas.models import LivenessResult, LivenessStatus

def test_verify_liveness_mock_pass():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("user_video_pass.mp4")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.PASSED
    assert res.confidence >= 0.90
    assert res.spoof_probability <= 0.10
    assert len(res.flags) == 0

def test_verify_liveness_mock_fail_spoof():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("user_video_spoof.mp4")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.FAILED
    assert res.confidence >= 0.85
    assert res.spoof_probability >= 0.80
    assert "no_blink_detected" in res.flags or "device_screen_glare" in res.flags

def test_verify_liveness_mock_fail_static():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("failed_video.mp4")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.FAILED
    assert res.spoof_probability >= 0.70

def test_verify_liveness_gesture_pass():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("user_video_pass.mp4", expected_gesture="2_fingers_near_eye")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.PASSED
    assert res.gestural_challenge_passed is True
    assert "gesture_2_fingers_near_eye_verified" in res.flags

def test_verify_liveness_gesture_fail():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("wrong_gesture_video.mp4", expected_gesture="2_fingers_near_eye")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.FAILED
    assert res.gestural_challenge_passed is False
    assert "gestural_challenge_failed" in res.flags

def test_verify_liveness_deepfake_fail():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("deepfake_spoof.mp4", expected_gesture="2_fingers_near_eye")
    assert isinstance(res, LivenessResult)
    assert res.liveness_status == LivenessStatus.FAILED
    assert res.digital_deepfake_detected is True
    assert "digital_deepfake_anomalies_detected" in res.flags
