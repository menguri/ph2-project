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


class PostSurveySubmit(BaseModel):
    participant_id: str
    episode_id: str
    fluency: Optional[int] = None
    contribution: Optional[int] = None
    trust: Optional[int] = None
    human_likeness: Optional[int] = None
    obstruction: Optional[int] = None
    frustration: Optional[int] = None
    play_again: Optional[int] = None
    open_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
