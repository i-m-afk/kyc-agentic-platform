from datetime import date
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

class LivenessStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ApplicationStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    ESCALATED = "ESCALATED"

class WatchlistHit(BaseModel):
    name: str
    list_name: str
    reason: str
    match_score: float

class AdverseMediaHit(BaseModel):
    title: str
    source: str
    sentiment: str
    url: Optional[str] = None

class OnboardingApplication(BaseModel):
    id: str
    id_image_path: str
    liveness_video_path: str
    status: ApplicationStatus = ApplicationStatus.PENDING
    created_at: str

class ExtractionResult(BaseModel):
    name: str = Field(..., min_length=1)
    dob: date
    id_number: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, v: date) -> date:
        if v >= date.today():
            raise ValueError("Date of birth must be in the past")
        return v

    @field_validator("id_number")
    @classmethod
    def validate_id_number(cls, v: str) -> str:
        if not any(c.isalnum() for c in v):
            raise ValueError("ID number must contain alphanumeric characters")
        return v

class LivenessResult(BaseModel):
    liveness_status: LivenessStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    spoof_probability: float = Field(..., ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)

class ScreeningResult(BaseModel):
    match_found: bool
    watchlist_hits: List[WatchlistHit] = Field(default_factory=list)
    adverse_media_hits: List[AdverseMediaHit] = Field(default_factory=list)
    risk_level: RiskLevel

class ConsolidatedRiskReport(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    explanation: str
    agent_audit_log: Dict[str, Any] = Field(default_factory=dict)
