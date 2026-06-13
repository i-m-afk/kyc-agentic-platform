import os
from datetime import date
import pytest
from src.orchestrator import run_kyc_pipeline
from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    LivenessStatus,
    ScreeningResult,
    RiskLevel,
    ConsolidatedRiskReport
)

def test_run_kyc_pipeline_clean_flow():
    os.environ["MOCK_ML"] = "True"
    
    # Run the pipeline with names that don't trigger mock watchlists/spoof rules
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="alice_smith_card.jpg",
        liveness_video_path="alice_smith_live.mp4"
    )
    
    # Assert return types
    assert isinstance(ext, ExtractionResult)
    assert isinstance(live, LivenessResult)
    assert isinstance(screen, ScreeningResult)
    assert isinstance(report, ConsolidatedRiskReport)
    
    # Assert data propagation
    assert ext.name == "Alice Smith"
    assert live.liveness_status == LivenessStatus.PASSED
    assert screen.match_found is False
    assert report.risk_level == RiskLevel.LOW
    
    # Assert audit log populated correctly
    assert "ExtractionAgent" in report.agent_audit_log
    assert "LivenessAgent" in report.agent_audit_log
    assert "ScreenerAgent" in report.agent_audit_log
    assert "RiskCoordinatorAgent" in report.agent_audit_log
    
    assert report.agent_audit_log["ExtractionAgent"]["status"] == "SUCCESS"
    assert report.agent_audit_log["LivenessAgent"]["status"] == "SUCCESS"

def test_run_kyc_pipeline_elevated_risk():
    os.environ["MOCK_ML"] = "True"
    
    # Run pipeline with a name that triggers a watchlist PEP hit and liveness spoof rules
    ext, live, screen, report = run_kyc_pipeline(
        id_image_path="john_doe_card.jpg",
        liveness_video_path="john_doe_spoof.mp4"
    )
    
    assert ext.name == "John Doe"
    assert live.liveness_status == LivenessStatus.FAILED
    assert screen.match_found is True
    assert report.risk_level == RiskLevel.HIGH
    assert report.risk_score >= 70
