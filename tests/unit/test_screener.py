import pytest
from datetime import date
from src.agents.screener import screen_applicant
from src.schemas.models import ScreeningResult, RiskLevel, ExtractionResult

def create_mock_extraction(name: str) -> ExtractionResult:
    return ExtractionResult(
        name=name,
        dob=date(1990, 1, 1),
        id_number="ID12345",
        confidence=0.95
    )

def test_screen_applicant_clean_low_risk():
    ext = create_mock_extraction("Alice Smith")
    res = screen_applicant(ext)
    assert isinstance(res, ScreeningResult)
    assert res.match_found is False
    assert len(res.watchlist_hits) == 0
    assert len(res.adverse_media_hits) == 0
    assert res.risk_level == RiskLevel.LOW

def test_screen_applicant_watchlist_high_risk():
    # Robert Vance matches Interpol Red Notice
    ext = create_mock_extraction("Robert Vance")
    res = screen_applicant(ext)
    assert isinstance(res, ScreeningResult)
    assert res.match_found is True
    assert len(res.watchlist_hits) > 0
    assert res.watchlist_hits[0].list_name == "Interpol Red Notice"
    assert res.risk_level == RiskLevel.HIGH

def test_screen_applicant_pep_medium_risk():
    # John Doe matches PEP under investigation
    ext = create_mock_extraction("John Doe")
    res = screen_applicant(ext)
    assert isinstance(res, ScreeningResult)
    assert res.match_found is True
    assert len(res.watchlist_hits) > 0
    assert res.watchlist_hits[0].list_name == "PEP (Politically Exposed Person)"
    assert res.risk_level == RiskLevel.MEDIUM

def test_screen_applicant_case_insensitive():
    ext = create_mock_extraction("jane doe")
    res = screen_applicant(ext)
    assert isinstance(res, ScreeningResult)
    assert res.match_found is True
    assert len(res.watchlist_hits) > 0
    assert res.risk_level == RiskLevel.HIGH

def test_screen_applicant_both_names():
    # Extracted name is clean, but submitted name is "jane doe" (matches watchlist)
    ext = create_mock_extraction("Alice Smith")
    res = screen_applicant(ext, applicant_name="Jane Doe")
    assert res.match_found is True
    assert any(hit.matched_on == "submitted_name" for hit in res.watchlist_hits)
