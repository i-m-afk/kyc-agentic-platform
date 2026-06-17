import os
import json
import httpx
import base64
import mimetypes
from typing import Optional, Dict, Any
from difflib import SequenceMatcher
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport
)
from src.utils.helpers import get_vllm_api_url, get_mock_ml_flag

def calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculates name similarity ratio using fuzzy sequence matching, with
    an adjustment for culturally common subset/middle name/father name inclusions,
    and penalties for mismatched first names.
    """
    if not name1 or not name2:
        return 0.0
        
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    
    # Base sequence matcher ratio
    ratio = SequenceMatcher(None, n1, n2).ratio()
    
    w1 = n1.split()
    w2 = n2.split()
    if len(w1) > 0 and len(w2) > 0:
        # Check first names similarity
        first_ratio = SequenceMatcher(None, w1[0], w2[0]).ratio()
        if first_ratio < 0.7:
            # Scale down the ratio because the primary/first name is different
            ratio = min(ratio, first_ratio)
            
        # Handle subset/token match (e.g. "Rishav Kumar" vs "Rishav Kumar Mishra")
        if len(w1) >= 2 and len(w2) >= 2:
            s1 = set(w1)
            s2 = set(w2)
            if s1.issubset(s2) or s2.issubset(s1):
                # If they share the first word/name, we boost the match score
                if w1[0] == w2[0]:
                    ratio = max(ratio, 0.85)
                
    return ratio


def get_base64_image_url(image_path: str) -> str:
    """
    Encodes an image to a base64 Data URL format.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        b64_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


def coordinate_risk(
    extraction: ExtractionResult,
    liveness: LivenessResult,
    screening: ScreeningResult,
    audit_log: dict,
    applicant_name: Optional[str] = None,
    id_image_path: Optional[str] = None,
    liveness_video_path: Optional[str] = None,
    aligned_id_image_path: Optional[str] = None
) -> ConsolidatedRiskReport:
    """
    Consolidates the outputs of the Extraction, Liveness, and Screener agents
    into a final risk score (0-100), risk level (LOW/MEDIUM/HIGH), and explanation.
    Uses a local vLLM VLM model (Qwen2-VL) to perform multimodal cognitive coordination
    when possible, inspecting raw visuals (aligned ID, face crops) to verify details
    and check for deepfakes/forgeries. Falls back to rule-based scoring if the server is offline.
    """
    # 1. Compute rule-based fallback values first (always available as fallback)
    score = 0.0
    factors = []

    # Liveness & Spoofing Risk
    if liveness.liveness_status == LivenessStatus.FAILED:
        liveness_score = 50.0 + (liveness.spoof_probability * 30.0)
        score += liveness_score
        
        if liveness.physical_spoof_detected:
            factors.append("Physical spoof detected (printed photo or screen replay)")
        if not liveness.gestural_challenge_passed:
            factors.append("Gestural challenge mismatch (incorrect finger/hand action)")
        if liveness.digital_deepfake_detected:
            factors.append("Digital deepfake/AI video anomalies detected")
            
        if not (liveness.physical_spoof_detected or not liveness.gestural_challenge_passed or liveness.digital_deepfake_detected):
            factors.append(f"Liveness check failed (spoof probability: {liveness.spoof_probability})")
    else:
        if liveness.spoof_probability > 0.15:
            score += liveness.spoof_probability * 20.0
            factors.append(f"Elevated spoof probability: {liveness.spoof_probability}")

    # Advanced Liveness Telemetry
    if liveness.fft_grid_detected:
        score += 40.0
        factors.append("Periodic frequency grid detected (indicative of digital replay/deepfake)")
        
    if not liveness.rppg_pulse_detected:
        score += 50.0
        factors.append("No physiological pulse detected (non-living print/screen presentation)")
        
    if liveness.optical_flow_mismatch:
        score += 35.0
        factors.append("Optical flow warping anomaly (face-swapping mask edge mismatch)")

    # Watchlist Screening
    if screening.match_found:
        if screening.risk_level == RiskLevel.HIGH:
            score += 65.0
            factors.append("Critical watchlist screening match detected (High risk)")
        elif screening.risk_level == RiskLevel.MEDIUM:
            score += 35.0
            factors.append("Watchlist or adverse media match detected (Medium risk)")
    
    # Extraction Confidence & Quality
    if extraction.confidence < 0.80:
        score += 15.0
        factors.append(f"Low document extraction confidence: {extraction.confidence}")
        
    if extraction.legibility_score < 0.70:
        score += 20.0
        factors.append(f"Low ID legibility/sharpness (blur detected, score: {extraction.legibility_score:.2f})")
        
    if not extraction.syntax_valid:
        score += 25.0
        factors.append("ID number format or check digit mismatch (DOB year/initials mismatch)")
        
    if not extraction.ovi_crest_detected:
        score += 15.0
        factors.append("Optically Variable Ink (OVI) hologram crest missing")

    # Identity fuzzy name match
    name_match_ratio = 1.0
    if applicant_name and extraction.name:
        name_match_ratio = calculate_name_similarity(applicant_name, extraction.name)
        if name_match_ratio < 0.80:
            score += 45.0
            factors.append(f"Identity mismatch: Submitted name '{applicant_name}' does not match ID name '{extraction.name}' (Match: {name_match_ratio*100:.1f}%)")

    # AI Generation / Forgery
    if extraction.forgery_detected or extraction.ai_generated_check in ("SUSPICIOUS", "AI_GENERATED"):
        score += 50.0
        reason = extraction.forgery_reason or "Suspicious textures or inconsistent fonts detected"
        factors.append(f"AI generation/forgery detected on ID image: {reason}")

    # Face verification match
    face_similarity = getattr(liveness, "face_similarity_score", 1.0)
    face_match_decision = getattr(liveness, "face_match_decision", "MATCH")
    if face_match_decision == "MISMATCH":
        score += 60.0
        factors.append(f"Face verification mismatch: ID photo and live face do not match (Similarity: {face_similarity*100:.1f}%)")
    elif face_similarity < 0.65:
        score += 30.0
        factors.append(f"Low face similarity score: ID photo and live face match is weak (Similarity: {face_similarity*100:.1f}%)")

    # Fallback/active indicators
    if getattr(extraction, "local_ocr_active", False):
        factors.append("Local EasyOCR fallback active (vLLM Qwen2-VL server was offline)")
    if getattr(liveness, "minifasnet_active", False):
        factors.append("Edge-friendly MiniFASNet model active for liveness detection")

    fallback_score = min(max(int(score), 0), 100)
    if (fallback_score >= 70 or 
        liveness.liveness_status == LivenessStatus.FAILED or 
        screening.risk_level == RiskLevel.HIGH or 
        extraction.forgery_detected or
        face_match_decision == "MISMATCH" or
        (applicant_name and extraction.name and name_match_ratio < 0.5)):
        fallback_level = RiskLevel.HIGH
    elif fallback_score >= 35 or screening.risk_level == RiskLevel.MEDIUM:
        fallback_level = RiskLevel.MEDIUM
    else:
        fallback_level = RiskLevel.LOW

    if not factors:
        fallback_explanation = "No risk factors detected. Applicant cleared."
    else:
        fallback_explanation = f"Risk factors identified: {'; '.join(factors)}."

    # Try Cognitive VLM Coordination if vLLM is available and MOCK_ML is false
    if not get_mock_ml_flag():
        api_url = get_vllm_api_url()
        # Query vLLM models registry
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        try:
            models_resp = httpx.get(f"{api_url}/models", timeout=2.0)
            if models_resp.status_code == 200:
                models_data = models_resp.json()
                if "data" in models_data and len(models_data["data"]) > 0:
                    model_name = models_data["data"][0]["id"]
        except Exception:
            pass

        system_prompt = (
            "You are a senior KYC compliance risk coordinator. Evaluate the following applicant telemetry "
            "and visually inspect the attached images (ID card, face cropped from ID, face cropped from live video). "
            "Pay critical attention to the live face crop, which is extracted from the active hand-face occlusion frame "
            "(the '3-Finger Test'). Inspect the boundaries where the fingers cross or overlay the face. Check for "
            "deepfake blending glitches: fingers warping into the face texture, digital face templates shifting or jittering "
            "under the hand, facial skin bleeding into fingers, or pixelation/blurring at occlusion seams. "
            "Decide the final application verdict. Reason through conflicting signals. "
            "Output EXACTLY in JSON format matching this schema:\n"
            "{\n"
            '  "risk_score": <0-100>,\n'
            '  "risk_level": "LOW"|"MEDIUM"|"HIGH",\n'
            '  "decision": "APPROVED"|"ESCALATED",\n'
            '  "explanation": "<reasoning>"\n'
            "}\n"
            "Do not include markdown fences or any other text before/after JSON."
        )

        user_prompt = f"""
Applicant Details:
- Submitted Name: {applicant_name or 'N/A'}
- Extracted Name from ID: {extraction.name or 'N/A'}
- Fuzzy Name Match Score: {name_match_ratio:.2f}

Biometric Telemetry:
- Face Verification Similarity Score: {face_similarity:.2f}
- Face Match Decision: {face_match_decision}

Liveness Telemetry:
- Liveness Status: {liveness.liveness_status.value}
- MiniFASNet Spoof Probability: {liveness.spoof_probability:.2f}
- MediaPipe Gesture Check: {'MATCH' if liveness.mediapipe_gesture_matched else 'MISMATCH'}
- Hand-Face Occlusion Ratio (3-Finger Test): {liveness.ensemble_metrics.get('gesture_occlusion_ratio', 0.0):.2f}
- FFT Peak Ratio (Compression/Deepfake check): {liveness.fft_metrics.get('peak_ratio', 1.4):.2f}
- rPPG Heartbeat check: {'PASS' if liveness.rppg_pulse_detected else 'FAIL'}
- Optical Flow movement: {'PASS' if not liveness.optical_flow_mismatch else 'FAIL'}

Compliance Screening:
- Watchlist Match: {'YES' if screening.match_found else 'NO'}
- Watchlist Risk Level: {screening.risk_level.value}

Document Quality:
- Extraction Confidence: {extraction.confidence:.2f}
- ID Legibility Score: {extraction.legibility_score:.2f}
- Hologram/Crest Detected: {'YES' if extraction.ovi_crest_detected else 'NO'}
- AI Generation/Forgery Detected: {'YES' if extraction.forgery_detected or extraction.ai_generated_check in ('SUSPICIOUS', 'AI_GENERATED') else 'NO'}
- Forgery Reason: {extraction.forgery_reason or 'None'}
"""

        user_content = [
            {"type": "text", "text": user_prompt}
        ]

        images_sent = []
        
        # Aligned/Raw ID Card image
        target_id_path = aligned_id_image_path or extraction.aligned_id_image_path or id_image_path
        if target_id_path and os.path.exists(target_id_path):
            try:
                b64_url = get_base64_image_url(target_id_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_url}
                })
                images_sent.append("ID Card Document (Aligned/Preprocessed)")
            except Exception as e:
                print(f"Failed to encode ID image: {e}")

        # We omit the cropped ID face to stay within vLLM's max 2-image constraint.
        # The VLM can verify likeness directly between the Aligned ID Card and the Cropped Live Face.

        # Crop Live Face
        live_face_path = getattr(liveness, "cropped_live_face_path", None)
        if live_face_path and os.path.exists(live_face_path):
            try:
                b64_url = get_base64_image_url(live_face_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": b64_url}
                })
                images_sent.append("Cropped Face from Live Video Frame")
            except Exception as e:
                print(f"Failed to encode live face crop: {e}")

        if images_sent:
            user_content[0]["text"] += "\nVisual Assets Sent for Analysis:\n" + "\n".join(f"- Image {i+1}: {desc}" for i, desc in enumerate(images_sent))

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 400
        }

        try:
            resp = httpx.post(f"{api_url}/chat/completions", json=payload, headers={"Content-Type": "application/json"}, timeout=15.0)
            if resp.status_code == 200:
                resp_data = resp.json()
                content = resp_data["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                llm_res = json.loads(content)
                risk_score = min(max(int(llm_res.get("risk_score", fallback_score)), 0), 100)
                risk_level_str = llm_res.get("risk_level", "HIGH").upper()
                if risk_level_str == "LOW":
                    risk_level = RiskLevel.LOW
                elif risk_level_str == "MEDIUM":
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.HIGH

                explanation = llm_res.get("explanation", fallback_explanation)
                
                return ConsolidatedRiskReport(
                    risk_score=risk_score,
                    risk_level=risk_level,
                    explanation=explanation,
                    coordinator_decision_mode="COGNITIVE_LLM",
                    agent_audit_log=audit_log
                )
        except Exception as e:
            print(f"vLLM coordination query failed: {e}. Falling back to rule-based coordination.")

    # Rule-based fallback return
    return ConsolidatedRiskReport(
        risk_score=fallback_score,
        risk_level=fallback_level,
        explanation=fallback_explanation,
        coordinator_decision_mode="RULE_BASED",
        agent_audit_log=audit_log
    )
