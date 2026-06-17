import os
from datetime import date
import pytest
from src.agents.extraction import extract_document_info
from src.schemas.models import ExtractionResult

def test_extract_document_info_mock_jane():
    os.environ["MOCK_ML"] = "True"
    # Even if file doesn't exist, mock extraction returns based on filename
    res = extract_document_info("jane_doe_id.jpg")
    assert isinstance(res, ExtractionResult)
    assert res.name == "Jane Doe"
    assert res.dob == date(1990, 5, 15)
    assert res.id_number == "JD9900515"
    assert res.confidence == 0.98

def test_extract_document_info_mock_john():
    os.environ["MOCK_ML"] = "True"
    res = extract_document_info("john_doe_card.png")
    assert isinstance(res, ExtractionResult)
    assert res.name == "John Doe"
    assert res.dob == date(1985, 11, 23)
    assert res.id_number == "JD851123X"
    assert res.confidence == 0.99

def test_extract_document_info_mock_robert():
    os.environ["MOCK_ML"] = "True"
    res = extract_document_info("robert_vance.jpg")
    assert isinstance(res, ExtractionResult)
    assert res.name == "Robert Vance"
    assert res.dob == date(1978, 2, 14)
    assert res.id_number == "RV780214"
    assert res.confidence == 0.95

def test_extract_document_info_mock_default():
    os.environ["MOCK_ML"] = "True"
    res = extract_document_info("some_other_id.jpg")
    assert isinstance(res, ExtractionResult)
    assert res.name == "Alice Smith"
    assert res.dob == date(1985, 11, 23)
    assert res.id_number == "A123456789"
    assert res.confidence == 0.90
    assert res.syntax_valid is True
    assert res.legibility_score >= 0.90

def test_extract_document_info_blurry():
    os.environ["MOCK_ML"] = "True"
    res = extract_document_info("blurry_id_card.jpg")
    assert isinstance(res, ExtractionResult)
    assert res.legibility_score <= 0.50

def test_id_syntax_validation():
    from src.agents.extraction import validate_id_syntax
    # Valid syntax matching DOB
    assert validate_id_syntax("AS950830", date(1995, 8, 30), "Alice Smith") is True
    assert validate_id_syntax("A123456789", date(1985, 11, 23), "Alice Smith") is True
    assert validate_id_syntax("RV780214", date(1978, 2, 14), "Robert Vance") is True
    # Invalid syntax due to mismatch in year/initials
    assert validate_id_syntax("JD9900515", date(1990, 5, 15), "Jane Doe") is False
    assert validate_id_syntax("XX123456", date(1995, 8, 30), "Alice Smith") is False

def test_extract_document_info_local_ocr_fallback():
    os.environ["MOCK_ML"] = "False"
    os.environ["VLLM_API_URL"] = "http://invalid-url-to-trigger-fallback.xyz"
    dummy_path = "jane_doe_temp_id.jpg"
    with open(dummy_path, "w") as f:
        f.write("dummy content")
    try:
        res = extract_document_info(dummy_path)
        assert isinstance(res, ExtractionResult)
        assert res.local_ocr_active is True
        assert res.name == "Jane Doe"
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

def test_indian_id_syntax_validation():
    from src.agents.extraction import validate_id_syntax
    # EPIC (Voter ID) format
    assert validate_id_syntax("RAB5212386", date(2001, 1, 18), "Rishav Kumar") is True
    # PAN Card format
    assert validate_id_syntax("ABCDE1234F", date(1995, 8, 30), "Alice Smith") is True
    # Aadhaar format
    assert validate_id_syntax("123456789012", date(1990, 5, 15), "Jane Doe") is True
