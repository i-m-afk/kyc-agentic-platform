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

def test_verify_liveness_advanced_telemetry():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("alice_smith_live.mp4")
    assert isinstance(res, LivenessResult)
    assert res.fft_grid_detected is False
    assert res.rppg_pulse_detected is True
    assert res.optical_flow_mismatch is False
    assert isinstance(res.fft_metrics, dict)
    assert "peak_ratio" in res.fft_metrics
    assert len(res.rppg_signal) > 0
    assert "mean_magnitude" in res.optical_flow_metrics

def test_fft_rppg_optical_flow_computations():
    import numpy as np
    from src.agents.liveness import compute_fft_metrics, compute_rppg_metrics, compute_optical_flow_metrics
    # Test with dummy frames
    dummy_frames = [np.ones((128, 128, 3), dtype=np.uint8) * 128 for _ in range(5)]
    
    fft_grid, fft_metrics = compute_fft_metrics(dummy_frames)
    assert fft_grid is False
    assert "peak_ratio" in fft_metrics
    
    rppg_pulse, rppg_signal = compute_rppg_metrics(dummy_frames)
    assert isinstance(rppg_pulse, bool)
    assert len(rppg_signal) > 0
    
    flow_mismatch, flow_metrics = compute_optical_flow_metrics(dummy_frames)
    assert flow_mismatch is False
    assert "mean_magnitude" in flow_metrics

def test_verify_liveness_face_similarity_match():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("alice_live.mp4", id_image_path="alice_id.jpg")
    assert res.face_match_decision == "MATCH"
    assert res.face_similarity_score >= 0.60
    assert "face_verification_mismatch" not in res.flags

def test_verify_liveness_face_similarity_mismatch():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("bob_mismatch.mp4", id_image_path="alice_id.jpg")
    assert res.face_match_decision == "MISMATCH"
    assert res.face_similarity_score < 0.60
    assert "face_verification_mismatch" in res.flags
    assert res.liveness_status == LivenessStatus.FAILED

def test_verify_liveness_minifasnet_active():
    os.environ["MOCK_ML"] = "True"
    res = verify_liveness("alice_live.mp4", use_minifasnet=True)
    assert res.minifasnet_active is True
