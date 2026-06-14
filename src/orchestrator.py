import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, Optional

from src.schemas.models import (
    ExtractionResult,
    LivenessResult,
    ScreeningResult,
    ConsolidatedRiskReport
)
from src.agents.extraction import extract_document_info
from src.agents.liveness import verify_liveness
from src.agents.screener import screen_identity
from src.agents.risk_coordinator import coordinate_risk
from src.utils.helpers import create_audit_entry, get_mock_ml_flag

def run_kyc_pipeline(
    id_image_path: str,
    liveness_video_path: str,
    expected_gesture: Optional[str] = None,
    applicant_name: Optional[str] = None
) -> Tuple[ExtractionResult, LivenessResult, ScreeningResult, ConsolidatedRiskReport]:
    """
    Coordinates the KYC processing pipeline:
    1. Runs Document Extraction and Liveness Verification in parallel.
    2. Runs Watchlist & Adverse Media Screening on the extracted name.
    3. Consolidates results into a single report.
    Tracks latency and model info in an audit log.
    """
    audit_log = {}
    
    # 1. Run parallel steps
    start_parallel = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        extraction_future = executor.submit(extract_document_info, id_image_path)
        liveness_future = executor.submit(verify_liveness, liveness_video_path, expected_gesture)
        
        try:
            extraction_res = extraction_future.result()
            extraction_latency = time.time() - start_parallel
            audit_log["ExtractionAgent"] = create_audit_entry(
                "ExtractionAgent",
                "SUCCESS",
                extraction_latency,
                "qwen2-vl" if not get_mock_ml_flag() else "mock_vision_rules",
                {"image_path": id_image_path}
            )
        except Exception as e:
            extraction_latency = time.time() - start_parallel
            audit_log["ExtractionAgent"] = create_audit_entry(
                "ExtractionAgent", "FAILED", extraction_latency, "unknown", {"error": str(e)}
            )
            raise e
            
        try:
            liveness_res = liveness_future.result()
            liveness_latency = time.time() - start_parallel
            audit_log["LivenessAgent"] = create_audit_entry(
                "LivenessAgent",
                "SUCCESS",
                liveness_latency,
                "mobilenet_v3_small" if not get_mock_ml_flag() else "mock_liveness_rules",
                {"video_path": liveness_video_path}
            )
        except Exception as e:
            liveness_latency = time.time() - start_parallel
            audit_log["LivenessAgent"] = create_audit_entry(
                "LivenessAgent", "FAILED", liveness_latency, "unknown", {"error": str(e)}
            )
            raise e

    # 2. Watchlist & Media Screening (depends on extracted name)
    start_screening = time.time()
    try:
        screening_res = screen_identity(extraction_res)
        screening_latency = time.time() - start_screening
        audit_log["ScreenerAgent"] = create_audit_entry(
            "ScreenerAgent",
            "SUCCESS",
            screening_latency,
            "mock_database_matcher",
            {"query_name": extraction_res.name}
        )
    except Exception as e:
        screening_latency = time.time() - start_screening
        audit_log["ScreenerAgent"] = create_audit_entry(
            "ScreenerAgent", "FAILED", screening_latency, "unknown", {"error": str(e)}
        )
        raise e

    # 3. Risk Coordination
    start_coordinator = time.time()
    try:
        risk_report = coordinate_risk(extraction_res, liveness_res, screening_res, audit_log, applicant_name)
        coordinator_latency = time.time() - start_coordinator
        audit_log["RiskCoordinatorAgent"] = create_audit_entry(
            "RiskCoordinatorAgent",
            "SUCCESS",
            coordinator_latency,
            "rule_based_risk_calculator",
            {}
        )
        risk_report.agent_audit_log = audit_log
    except Exception as e:
        coordinator_latency = time.time() - start_coordinator
        audit_log["RiskCoordinatorAgent"] = create_audit_entry(
            "RiskCoordinatorAgent", "FAILED", coordinator_latency, "unknown", {"error": str(e)}
        )
        raise e

    return extraction_res, liveness_res, screening_res, risk_report
