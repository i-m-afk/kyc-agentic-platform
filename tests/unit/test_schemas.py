from datetime import date, timedelta
import pytest
from pydantic import ValidationError
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport,
    WatchlistHit,
    AdverseMediaHit
)

def test_extraction_result_valid():
    dob = date.today() - timedelta(days=365 * 25)
    res = ExtractionResult(
        name="John Doe",
        dob=dob,
        id_number="AB-12345-XY",
        confidence=0.95
    )
    assert res.name == "John Doe"
    assert res.dob == dob
    assert res.id_number == "AB-12345-XY"
    assert res.confidence == 0.95

def test_extraction_result_invalid_name():
    dob = date.today() - timedelta(days=365 * 25)
    with pytest.raises(ValidationError):
        ExtractionResult(
            name="",
            dob=dob,
            id_number="AB12345XY",
            confidence=0.95
        )

def test_extraction_result_invalid_dob():
    future_dob = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError):
        ExtractionResult(
            name="John Doe",
            dob=future_dob,
            id_number="AB12345XY",
            confidence=0.95
        )

def test_extraction_result_invalid_id_number():
    dob = date.today() - timedelta(days=365 * 25)
    with pytest.raises(ValidationError):
        ExtractionResult(
            name="John Doe",
            dob=dob,
            id_number="---!!!",
            confidence=0.95
        )

def test_liveness_result_valid():
    res = LivenessResult(
        liveness_status=LivenessStatus.PASSED,
        confidence=0.99,
        spoof_probability=0.01,
        flags=[]
    )
    assert res.liveness_status == LivenessStatus.PASSED
    assert res.confidence == 0.99
    assert res.spoof_probability == 0.01

def test_screening_result_valid():
    hit1 = WatchlistHit(name="John Doe", list_name="OFAC", reason="Match", match_score=0.85)
    hit2 = AdverseMediaHit(title="Scam Investigation", source="News", sentiment="Negative")
    res = ScreeningResult(
        match_found=True,
        watchlist_hits=[hit1],
        adverse_media_hits=[hit2],
        risk_level=RiskLevel.MEDIUM
    )
    assert res.match_found is True
    assert len(res.watchlist_hits) == 1
    assert res.risk_level == RiskLevel.MEDIUM

def test_consolidated_risk_report_valid():
    report = ConsolidatedRiskReport(
        risk_score=75,
        risk_level=RiskLevel.HIGH,
        explanation="Due to watchlists match",
        agent_audit_log={"extraction": {"time": 0.5}}
    )
    assert report.risk_score == 75
    assert report.risk_level == RiskLevel.HIGH

def test_consolidated_risk_report_invalid_score():
    with pytest.raises(ValidationError):
        ConsolidatedRiskReport(
            risk_score=105,
            risk_level=RiskLevel.HIGH,
            explanation="Invalid score",
            agent_audit_log={}
        )
