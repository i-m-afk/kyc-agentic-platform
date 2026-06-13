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
    audit_log: dict
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

    # Clamp score
    final_score = min(max(int(score), 0), 100)

    # Determine risk level
    if final_score >= 70 or liveness.liveness_status == LivenessStatus.FAILED or screening.risk_level == RiskLevel.HIGH:
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
