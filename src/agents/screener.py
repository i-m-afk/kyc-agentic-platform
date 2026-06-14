from typing import Optional
from src.schemas.models import (
    ScreeningResult,
    RiskLevel,
    WatchlistHit,
    AdverseMediaHit,
    ExtractionResult
)
from src.utils.db import search_watchlist, search_adverse_media

def screen_applicant(extracted_info: ExtractionResult, applicant_name: Optional[str] = None) -> ScreeningResult:
    """
    Screens candidate's name against PEP/Sanction watchlists and adverse media databases.
    Categorizes the risk level as LOW, MEDIUM, or HIGH.
    """
    extracted_name = extracted_info.name
    query_names = []
    
    # 1. Screen the extracted name from the ID card
    if extracted_name:
        query_names.append((extracted_name, "extracted_name"))
        
    # 2. Also screen the submitted applicant name if it's different
    if applicant_name and applicant_name.lower().strip() != extracted_name.lower().strip():
        query_names.append((applicant_name, "submitted_name"))
    
    watchlist_hits = []
    adverse_media_hits = []
    
    for name, matched_on in query_names:
        watchlist_raw = search_watchlist(name)
        adverse_media_raw = search_adverse_media(name)
        
        for h in watchlist_raw:
            watchlist_hits.append(
                WatchlistHit(
                    name=h["name"],
                    list_name=h["list_name"],
                    reason=h["reason"],
                    match_score=h["match_score"],
                    matched_on=matched_on
                )
            )
            
        for h in adverse_media_raw:
            adverse_media_hits.append(
                AdverseMediaHit(
                    title=h["title"],
                    source=h["source"],
                    sentiment=h["sentiment"],
                    url=h.get("url"),
                    matched_on=matched_on
                )
            )
    
    match_found = len(watchlist_hits) > 0 or len(adverse_media_hits) > 0
    
    # Risk decision logic
    if watchlist_hits:
        # Critical lists trigger HIGH risk
        is_critical = any(
            hit.list_name in ["Interpol Red Notice", "OFAC Sanctions List"] and hit.match_score >= 0.7
            for hit in watchlist_hits
        )
        risk_level = RiskLevel.HIGH if is_critical else RiskLevel.MEDIUM
    elif adverse_media_hits:
        risk_level = RiskLevel.MEDIUM
    else:
        risk_level = RiskLevel.LOW
        
    return ScreeningResult(
        match_found=match_found,
        watchlist_hits=watchlist_hits,
        adverse_media_hits=adverse_media_hits,
        risk_level=risk_level
    )

screen_identity = screen_applicant

