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
    collisions = Column(Integer, default=0)
    deliveries = Column(Integer, default=0)
    trajectory_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    participant = relationship("Participant", back_populates="episodes")
    survey_response = relationship("SurveyResponse", back_populates="episode", uselist=False)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id = Column(String, primary_key=True)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    fluency = Column(Integer, nullable=True)
    contribution = Column(Integer, nullable=True)
    trust = Column(Integer, nullable=True)
    human_likeness = Column(Integer, nullable=True)
    obstruction = Column(Integer, nullable=True)
    frustration = Column(Integer, nullable=True)
    play_again = Column(Integer, nullable=True)
    open_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    episode = relationship("Episode", back_populates="survey_response")
