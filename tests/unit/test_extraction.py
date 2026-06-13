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
    assert res.dob == date(1995, 8, 30)
    assert res.id_number == "AS950830"
    assert res.confidence == 0.90
