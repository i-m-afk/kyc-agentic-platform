import os
from src.utils.helpers import get_config, get_mock_ml_flag, create_audit_entry
from src.utils.db import search_watchlist, search_adverse_media

def test_get_config_env():
    os.environ["TEST_ENV_VAR"] = "True"
    assert get_config("TEST_ENV_VAR", False) is True
    
    os.environ["TEST_ENV_VAR_INT"] = "42"
    assert get_config("TEST_ENV_VAR_INT", 0) == 42
    
    os.environ["TEST_ENV_VAR_STR"] = "Hello"
    assert get_config("TEST_ENV_VAR_STR", "") == "Hello"

def test_get_mock_ml_flag_default():
    if "MOCK_ML" in os.environ:
        del os.environ["MOCK_ML"]
    assert get_mock_ml_flag() is True

def test_create_audit_entry():
    entry = create_audit_entry("test_agent", "SUCCESS", 1.25, "mock_model", {"key": "val"})
    assert entry["status"] == "SUCCESS"
    assert entry["latency_seconds"] == 1.25
    assert entry["model_used"] == "mock_model"
    assert entry["details"] == {"key": "val"}
    assert "timestamp" in entry

def test_search_watchlist_hits():
    # Exact match
    hits = search_watchlist("John Doe")
    assert len(hits) == 1
    assert hits[0]["list_name"] == "PEP (Politically Exposed Person)"
    assert hits[0]["match_score"] == 1.0

    # Substring match
    hits_sub = search_watchlist("Vance")
    assert len(hits_sub) == 1
    assert hits_sub[0]["name"] == "Robert Vance"
    assert hits_sub[0]["match_score"] >= 0.5

    # No match
    hits_empty = search_watchlist("Unrelated Person")
    assert len(hits_empty) == 0

def test_search_adverse_media():
    hits = search_adverse_media("Jane Doe")
    assert len(hits) == 1
    assert "Unreported Off-shore Assets" in hits[0]["title"]

    hits_empty = search_adverse_media("Unknown Person")
    assert len(hits_empty) == 0
