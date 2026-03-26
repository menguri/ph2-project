"""
Phase 7: Database session management.
"""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


_engine = None
_SessionLocal = None


def init_db(db_path: str):
    global _engine, _SessionLocal
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", echo=False)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)


def get_session():
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_sync():
    """Non-generator version for simple use."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized.")
    return _SessionLocal()
