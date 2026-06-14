from typing import Optional
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport
)

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

    # 1. Liveness Risk
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

    # 2. Watchlist Screening Risk
    if screening.match_found:
        if screening.risk_level == RiskLevel.HIGH:
            score += 65.0
            factors.append("Critical watchlist screening match detected (High risk)")
        elif screening.risk_level == RiskLevel.MEDIUM:
            score += 35.0
            factors.append("Watchlist or adverse media match detected (Medium risk)")
    
    # 3. Extraction Confidence Risk
    if extraction.confidence < 0.80:
        score += 15.0
        factors.append(f"Low document extraction confidence: {extraction.confidence}")

    # 4. Identity & Name Matching Risk (Fuzzy matching)
    if applicant_name and extraction.name:
        from difflib import SequenceMatcher
        name1 = applicant_name.lower().strip()
        name2 = extraction.name.lower().strip()
        match_ratio = SequenceMatcher(None, name1, name2).ratio()
        if match_ratio < 0.80:
            score += 45.0
            factors.append(f"Identity mismatch: Submitted name '{applicant_name}' does not match ID name '{extraction.name}' (Match: {match_ratio*100:.1f}%)")

    # 5. AI Generation & Digital Forgery Risk
    if extraction.forgery_detected or extraction.ai_generated_check in ("SUSPICIOUS", "AI_GENERATED"):
        score += 50.0
        reason = extraction.forgery_reason or "Suspicious textures or inconsistent fonts detected"
        factors.append(f"AI generation/forgery detected on ID image: {reason}")

    # Clamp score
    final_score = min(max(int(score), 0), 100)

    # Determine risk level
    if (final_score >= 70 or 
        liveness.liveness_status == LivenessStatus.FAILED or 
        screening.risk_level == RiskLevel.HIGH or 
        extraction.forgery_detected or
        (applicant_name and extraction.name and SequenceMatcher(None, applicant_name.lower().strip(), extraction.name.lower().strip()).ratio() < 0.5)):
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
