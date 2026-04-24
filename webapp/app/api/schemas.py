"""
Phase 6: Pydantic schemas.
"""
from pydantic import BaseModel
from typing import Optional


class PreSurveySubmit(BaseModel):
    participant_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    gaming_exp: Optional[int] = None
    overcooked_exp: Optional[str] = None
    consent_18plus: Optional[bool] = None
    consent_voluntary: Optional[bool] = None
    consent_timestamp: Optional[str] = None


class PostSurveySubmit(BaseModel):
    participant_id: str
    episode_id: str
    # 8개 주관 지표 (H2/H3 대응)
    adaptive: Optional[int] = None
    consistent: Optional[int] = None
    human_like: Optional[int] = None
    in_my_way: Optional[int] = None        # 역채점
    frustrating: Optional[int] = None      # 역채점
    enjoyed: Optional[int] = None
    coordination: Optional[int] = None     # 핵심 H2
    workload: Optional[int] = None         # NASA-TLX (부담↓=좋음)
    open_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
