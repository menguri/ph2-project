"""
Phase 7: SQLAlchemy ORM models.
"""
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Participant(Base):
    __tablename__ = "participants"

    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    pre_survey = Column(JSON, nullable=True)

    episodes = relationship("Episode", back_populates="participant")


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(String, primary_key=True)
    participant_id = Column(String, ForeignKey("participants.id"), nullable=False)
    algo_name = Column(String, nullable=False)
    seed_id = Column(String, nullable=False)
    layout = Column(String, nullable=False)
    human_player_index = Column(Integer, nullable=False)
    final_score = Column(Integer, default=0)
    episode_length = Column(Integer, default=0)
    # 객관 지표 (H1/H3/H4 대응)
    collisions = Column(Integer, default=0)                    # 충돌 횟수 (H3 핵심)
    deliveries = Column(Integer, default=0)                    # 순수 배달 성공 수 (H1)
    role_specialization = Column(Float, nullable=True)         # 역할 분화 지수 0~1 (H4 핵심)
    idle_time_ratio = Column(Float, nullable=True)             # AI 비생산 timestep 비율 (H1 sanity)
    # per-task 세부 카운트 (JSON): {"onion_pickup": [h,a], "pot_deposit": [h,a], ...}
    task_events = Column(JSON, nullable=True)
    trajectory_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    participant = relationship("Participant", back_populates="episodes")
    survey_response = relationship("SurveyResponse", back_populates="episode", uselist=False)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id = Column(String, primary_key=True)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    # 8개 주관 지표 (1-7 Likert, workload 만 Very Low~Very High)
    adaptive = Column(Integer, nullable=True)
    consistent = Column(Integer, nullable=True)
    human_like = Column(Integer, nullable=True)
    in_my_way = Column(Integer, nullable=True)          # 역채점
    frustrating = Column(Integer, nullable=True)        # 역채점
    enjoyed = Column(Integer, nullable=True)
    coordination = Column(Integer, nullable=True)       # 핵심 H2
    workload = Column(Integer, nullable=True)           # NASA-TLX
    open_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    episode = relationship("Episode", back_populates="survey_response")
