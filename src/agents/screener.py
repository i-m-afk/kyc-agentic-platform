from src.schemas.models import (
    ScreeningResult,
    RiskLevel,
    WatchlistHit,
    AdverseMediaHit,
    ExtractionResult
)
from src.utils.db import search_watchlist, search_adverse_media

def screen_applicant(extracted_info: ExtractionResult) -> ScreeningResult:
    """
    Screens candidate's name against PEP/Sanction watchlists and adverse media databases.
    Categorizes the risk level as LOW, MEDIUM, or HIGH.
    """
    name = extracted_info.name
    watchlist_raw = search_watchlist(name)
    adverse_media_raw = search_adverse_media(name)
    
    watchlist_hits = [
        WatchlistHit(
            name=h["name"],
            list_name=h["list_name"],
            reason=h["reason"],
            match_score=h["match_score"]
        )
        for h in watchlist_raw
    ]
    
    adverse_media_hits = [
        AdverseMediaHit(
            title=h["title"],
            source=h["source"],
            sentiment=h["sentiment"],
            url=h.get("url")
        )
        for h in adverse_media_raw
    ]
    
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

