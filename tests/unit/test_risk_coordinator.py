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
