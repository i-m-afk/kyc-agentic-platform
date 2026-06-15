import pytest
from datetime import date
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport
)
from src.agents.risk_coordinator import coordinate_risk

def test_coordinate_risk_low_clean():
    ext = ExtractionResult(name="Alice Smith", dob=date(1995, 8, 30), id_number="AS950830", confidence=0.95)
    live = LivenessResult(liveness_status=LivenessStatus.PASSED, confidence=0.98, spoof_probability=0.02, flags=[])
    screen = ScreeningResult(match_found=False, watchlist_hits=[], adverse_media_hits=[], risk_level=RiskLevel.LOW)
    
    report = coordinate_risk(ext, live, screen, audit_log={})
    assert isinstance(report, ConsolidatedRiskReport)
    assert report.risk_level == RiskLevel.LOW
    assert report.risk_score <= 35
    assert "No risk factors detected" in report.explanation

def test_coordinate_risk_high_watchlist():
    ext = ExtractionResult(name="Jane Doe", dob=date(1990, 5, 15), id_number="JD9900515", confidence=0.98)
    live = LivenessResult(liveness_status=LivenessStatus.PASSED, confidence=0.97, spoof_probability=0.03, flags=[])
    # Screen has high risk due to OFAC match
    screen = ScreeningResult(match_found=True, watchlist_hits=[], adverse_media_hits=[], risk_level=RiskLevel.HIGH)
    
    report = coordinate_risk(ext, live, screen, audit_log={})
    assert report.risk_level == RiskLevel.HIGH
    assert report.risk_score >= 60
    assert "watchlist match" in report.explanation.lower() or "screening" in report.explanation.lower()

def test_coordinate_risk_failed_liveness():
    ext = ExtractionResult(name="Alice Smith", dob=date(1995, 8, 30), id_number="AS950830", confidence=0.95)
    # Failed liveness check
    live = LivenessResult(liveness_status=LivenessStatus.FAILED, confidence=0.92, spoof_probability=0.88, flags=["device_screen_glare"])
    screen = ScreeningResult(match_found=False, watchlist_hits=[], adverse_media_hits=[], risk_level=RiskLevel.LOW)
    
    report = coordinate_risk(ext, live, screen, audit_log={})
    assert report.risk_level == RiskLevel.HIGH
    assert report.risk_score >= 70
    assert "liveness check failed" in report.explanation.lower()

def test_calculate_name_similarity():
    from src.agents.risk_coordinator import calculate_name_similarity
    # Subset match (first names match, and one is a subset of the other)
    assert calculate_name_similarity("Rishav Kumar", "Rishav Kumar Mishra") >= 0.85
    # Sibling or parent mismatch (e.g. same surname/middle name but different first name)
    assert calculate_name_similarity("Rakesh Kumar Mishra", "Rishav Kumar Mishra") < 0.80
    # Completely different names
    assert calculate_name_similarity("Alice Smith", "Bob Miller") < 0.50

def test_coordinate_risk_fuzzy_name_match():
    ext = ExtractionResult(name="Rishav Kumar Mishra", dob=date(2001, 1, 18), id_number="RAB5212386", confidence=0.95, syntax_valid=True)
    live = LivenessResult(liveness_status=LivenessStatus.PASSED, confidence=0.98, spoof_probability=0.02, flags=[])
    screen = ScreeningResult(match_found=False, watchlist_hits=[], adverse_media_hits=[], risk_level=RiskLevel.LOW)
    
    # applicant_name matches the subset "Rishav Kumar"
    report = coordinate_risk(ext, live, screen, audit_log={}, applicant_name="Rishav Kumar")
    assert report.risk_level == RiskLevel.LOW
    assert "Identity mismatch" not in report.explanation

def test_coordinate_risk_decision_mode():
    ext = ExtractionResult(name="Alice Smith", dob=date(1995, 8, 30), id_number="AS950830", confidence=0.95)
    live = LivenessResult(liveness_status=LivenessStatus.PASSED, confidence=0.98, spoof_probability=0.02, flags=[])
    screen = ScreeningResult(match_found=False, watchlist_hits=[], adverse_media_hits=[], risk_level=RiskLevel.LOW)
    
    # Under MOCK_ML, decision mode must be RULE_BASED fallback
    report = coordinate_risk(ext, live, screen, audit_log={})
    assert report.coordinator_decision_mode == "RULE_BASED"
