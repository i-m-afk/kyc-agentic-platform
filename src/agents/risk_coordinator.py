from typing import Optional
from difflib import SequenceMatcher
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport
)

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


def coordinate_risk(
    extraction: ExtractionResult,
    liveness: LivenessResult,
    screening: ScreeningResult,
    audit_log: dict,
    applicant_name: Optional[str] = None
) -> ConsolidatedRiskReport:
    """
    Consolidates the outputs of the Extraction, Liveness, and Screener agents
    into a final risk score (0-100), risk level (LOW/MEDIUM/HIGH), and explanation.
    """
    score = 0.0
    factors = []

    # 1. Liveness & Spoofing Risk
    if liveness.liveness_status == LivenessStatus.FAILED:
        # High penalty for failed liveness
        liveness_score = 50.0 + (liveness.spoof_probability * 30.0)
        score += liveness_score
        
        # Add specific factors based on the detailed spoof detections
        if liveness.physical_spoof_detected:
            factors.append("Physical spoof detected (printed photo or screen replay)")
        if not liveness.gestural_challenge_passed:
            factors.append("Gestural challenge mismatch (incorrect finger/hand action)")
        if liveness.digital_deepfake_detected:
            factors.append("Digital deepfake/AI video anomalies detected")
            
        if not (liveness.physical_spoof_detected or not liveness.gestural_challenge_passed or liveness.digital_deepfake_detected):
            factors.append(f"Liveness check failed (spoof probability: {liveness.spoof_probability})")
    else:
        # Small contribution for low confidence liveness
        if liveness.spoof_probability > 0.15:
            score += liveness.spoof_probability * 20.0
            factors.append(f"Elevated spoof probability: {liveness.spoof_probability}")

    # 2. Advanced Mathematical Liveness Telemetry Penalties
    if liveness.fft_grid_detected:
        score += 40.0
        factors.append("Periodic frequency grid detected (indicative of digital replay/deepfake)")
        
    if not liveness.rppg_pulse_detected:
        score += 50.0
        factors.append("No physiological pulse detected (non-living print/screen presentation)")
        
    if liveness.optical_flow_mismatch:
        score += 35.0
        factors.append("Optical flow warping anomaly (face-swapping mask edge mismatch)")

    # 3. Watchlist Screening Risk
    if screening.match_found:
        if screening.risk_level == RiskLevel.HIGH:
            score += 65.0
            factors.append("Critical watchlist screening match detected (High risk)")
        elif screening.risk_level == RiskLevel.MEDIUM:
            score += 35.0
            factors.append("Watchlist or adverse media match detected (Medium risk)")
    
    # 4. Extraction Confidence & Quality Risk
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

    # 5. Identity & Name Matching Risk (Fuzzy matching)
    if applicant_name and extraction.name:
        match_ratio = calculate_name_similarity(applicant_name, extraction.name)
        if match_ratio < 0.80:
            score += 45.0
            factors.append(f"Identity mismatch: Submitted name '{applicant_name}' does not match ID name '{extraction.name}' (Match: {match_ratio*100:.1f}%)")

    # 6. AI Generation & Digital Forgery Risk
    if extraction.forgery_detected or extraction.ai_generated_check in ("SUSPICIOUS", "AI_GENERATED"):
        score += 50.0
        reason = extraction.forgery_reason or "Suspicious textures or inconsistent fonts detected"
        factors.append(f"AI generation/forgery detected on ID image: {reason}")

    # 7. Face Verification Match
    if getattr(liveness, "face_match_decision", "MATCH") == "MISMATCH":
        score += 60.0
        similarity = getattr(liveness, "face_similarity_score", 0.0)
        factors.append(f"Face verification mismatch: ID photo and live face do not match (Similarity: {similarity*100:.1f}%)")
    elif getattr(liveness, "face_similarity_score", 1.0) < 0.65:
        score += 30.0
        similarity = getattr(liveness, "face_similarity_score", 1.0)
        factors.append(f"Low face similarity score: ID photo and live face match is weak (Similarity: {similarity*100:.1f}%)")

    # 8. Fallback and model active indicators
    if getattr(extraction, "local_ocr_active", False):
        factors.append("Local EasyOCR fallback active (vLLM Qwen2-VL server was offline)")
    if getattr(liveness, "minifasnet_active", False):
        factors.append("Edge-friendly MiniFASNet model active for liveness detection")

    # Clamp score
    final_score = min(max(int(score), 0), 100)

    # Determine risk level
    if (final_score >= 70 or 
        liveness.liveness_status == LivenessStatus.FAILED or 
        screening.risk_level == RiskLevel.HIGH or 
        extraction.forgery_detected or
        getattr(liveness, "face_match_decision", "MATCH") == "MISMATCH" or
        (applicant_name and extraction.name and calculate_name_similarity(applicant_name, extraction.name) < 0.5)):
        final_level = RiskLevel.HIGH
    elif final_score >= 35 or screening.risk_level == RiskLevel.MEDIUM:
        final_level = RiskLevel.MEDIUM
    else:
        final_level = RiskLevel.LOW

    # Generate explanation
    if not factors:
        explanation = "No risk factors detected. Applicant cleared."
    else:
        explanation = f"Risk factors identified: {'; '.join(factors)}."

    return ConsolidatedRiskReport(
        risk_score=final_score,
        risk_level=final_level,
        explanation=explanation,
        agent_audit_log=audit_log
    )
