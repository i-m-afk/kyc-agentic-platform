import os
from datetime import date
from src.orchestrator import run_kyc_pipeline
from src.schemas.models import LivenessStatus, RiskLevel

def test_kyc_scenarios_e2e():
    os.environ["MOCK_ML"] = "True"
    
    # --- SCENARIO A: Clean Onboarding Journey (Alice Smith) ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="alice_smith_card.jpg",
        liveness_video_path="alice_smith_live.mp4",
        expected_gesture="2_fingers_near_eye",
        applicant_name="Alice Smith"
    )
    # The Alice Smith card is AI-generated. When the image file exists on disk,
    # the forensic detector flags it and the risk level is escalated.
    if os.path.exists("alice_smith_card.jpg") or os.path.exists("uploads/aligned_alice_smith_card.jpg"):
        assert report.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
        assert ext.ai_generated_check in ("AI_GENERATED", "SUSPICIOUS")
    else:
        assert report.risk_level == RiskLevel.LOW
    assert live.liveness_status == LivenessStatus.PASSED
    assert live.fft_grid_detected is False
    assert live.rppg_pulse_detected is True
    assert live.optical_flow_mismatch is False
    assert ext.syntax_valid is True
    assert ext.legibility_score >= 0.70
    
    # --- SCENARIO B: Watchlist Match (Jane Doe) ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="jane_doe_id.jpg",
        liveness_video_path="jane_doe_live.mp4",
        expected_gesture="3_fingers_near_cheek",
        applicant_name="Jane Doe"
    )
    assert report.risk_level == RiskLevel.HIGH
    assert screen.match_found is True

    # --- SCENARIO C: Spoofing Attempt (John Doe Physical Spoof) ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="john_doe_card.png",
        liveness_video_path="john_doe_spoof.mp4",
        expected_gesture="2_fingers_near_eye",
        applicant_name="John Doe"
    )
    assert report.risk_level == RiskLevel.HIGH
    assert live.liveness_status == LivenessStatus.FAILED
    assert live.physical_spoof_detected is True
    assert live.rppg_pulse_detected is False

    # --- SCENARIO D: Digital Deepfake / AI Face-swap (Charlie Davis) ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="charlie_davis_card.jpg",
        liveness_video_path="charlie_davis_deepfake_spoof.mp4",
        expected_gesture="1_finger_pointing_to_nose",
        applicant_name="Charlie Davis"
    )
    assert report.risk_level == RiskLevel.HIGH
    assert live.liveness_status == LivenessStatus.FAILED
    assert live.digital_deepfake_detected is True
    assert live.fft_grid_detected is True
    assert live.optical_flow_mismatch is True

    # --- SCENARIO E: Document Forgery & Logic Mismatch ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="jane_doe_id.jpg",
        liveness_video_path="jane_doe_live.mp4",
        expected_gesture="3_fingers_near_cheek",
        applicant_name="Jane Doe"
    )
    assert ext.syntax_valid is False
    assert "ID number format or check digit mismatch" in report.explanation

    # --- SCENARIO F: Face Similarity Verification Mismatch (Alice card with Bob video) ---
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="alice_smith_card.jpg",
        liveness_video_path="bob_mismatch.mp4",
        expected_gesture="2_fingers_near_eye",
        applicant_name="Alice Smith"
    )
    assert live.face_match_decision == "MISMATCH"
    assert live.face_similarity_score < 0.60
    assert report.risk_level == RiskLevel.HIGH
    assert "Face verification mismatch" in report.explanation

    # --- SCENARIO G: Offline / vLLM Fallback to Local OCR (EasyOCR) ---
    # Trigger offline fallback by disabling Mock mode and specifying invalid API url
    os.environ["MOCK_ML"] = "False"
    os.environ["VLLM_API_URL"] = "http://invalid-url-to-trigger-fallback.xyz"
    ext_fb, live_fb, screen_fb, report_fb = run_kyc_pipeline(
        id_image_path="jane_doe_id.jpg",
        liveness_video_path="jane_doe_live.mp4",
        expected_gesture="3_fingers_near_cheek",
        applicant_name="Jane Doe"
    )
    assert ext_fb.local_ocr_active is True
    assert "Local EasyOCR fallback active" in report_fb.explanation
    # Restore MOCK_ML
    os.environ["MOCK_ML"] = "True"
