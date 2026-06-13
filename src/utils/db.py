from typing import List, Dict, Any

# Mock databases
MOCK_WATCHLIST = [
    {
        "name": "Jane Doe",
        "list_name": "OFAC Sanctions List",
        "reason": "Suspected financial crime involvement",
        "match_score": 1.0
    },
    {
        "name": "John Doe",
        "list_name": "PEP (Politically Exposed Person)",
        "reason": "Former government official under investigation",
        "match_score": 1.0
    },
    {
        "name": "Robert Vance",
        "list_name": "Interpol Red Notice",
        "reason": "Money laundering operations",
        "match_score": 1.0
    }
]

MOCK_ADVERSE_MEDIA = [
    {
        "name": "Jane Doe",
        "title": "Doe Enterprises Under Audit for Unreported Off-shore Assets",
        "source": "Global Financial Tribune",
        "sentiment": "Negative",
        "url": "https://example.com/news/doe-audit"
    },
    {
        "name": "Robert Vance",
        "title": "Shell Company Network Linked to Robert Vance Exposed",
        "source": "Investigative Leak Hub",
        "sentiment": "Negative",
        "url": "https://example.com/news/vance-expose"
    },
    {
        "name": "John Doe",
        "title": "Local Politician John Doe Announces Resignation Amid Probe",
        "source": "Daily Sentinel",
        "sentiment": "Negative",
        "url": "https://example.com/news/doe-resigns"
    }
]

def search_watchlist(name: str) -> List[Dict[str, Any]]:
    """Searches mock watchlist database for matches using substring/case-insensitive similarity."""
    cleaned_query = name.strip().lower()
    if not cleaned_query:
        return []
    
    hits = []
    for entry in MOCK_WATCHLIST:
        entry_name = entry["name"].lower()
        if cleaned_query in entry_name or entry_name in cleaned_query:
            # Simple match score calculation
            score = round(min(len(cleaned_query), len(entry_name)) / max(len(cleaned_query), len(entry_name)), 2)
            score = min(max(score, 0.5), 1.0)
            hit = entry.copy()
            hit["match_score"] = score
            hits.append(hit)
    return hits

def search_adverse_media(name: str) -> List[Dict[str, Any]]:
    """Searches mock adverse media database for matching articles."""
    cleaned_query = name.strip().lower()
    if not cleaned_query:
        return []
        
    hits = []
    for entry in MOCK_ADVERSE_MEDIA:
        entry_name = entry["name"].lower()
        if cleaned_query in entry_name or entry_name in cleaned_query:
            hits.append(entry.copy())
    return hits
