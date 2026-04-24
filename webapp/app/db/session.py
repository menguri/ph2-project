"""
Phase 7: Database session management.
"""
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from .models import Base


# 기존 DB 에 새 컬럼을 추가하기 위한 간단한 마이그레이션 테이블.
# (column_name, ddl_type) — ALTER TABLE ADD COLUMN 으로 nullable 컬럼만 추가 가능.
_MIGRATIONS = {
    "episodes": [
        ("role_specialization", "FLOAT"),
        ("idle_time_ratio", "FLOAT"),
        ("task_events", "JSON"),
    ],
    "survey_responses": [
        ("adaptive", "INTEGER"),
        ("consistent", "INTEGER"),
        ("human_like", "INTEGER"),
        ("in_my_way", "INTEGER"),
        ("frustrating", "INTEGER"),
        ("enjoyed", "INTEGER"),
        ("coordination", "INTEGER"),
        ("workload", "INTEGER"),
    ],
}


def _apply_migrations(engine):
    """기존 SQLite 테이블에 빠진 컬럼이 있으면 ALTER TABLE ADD COLUMN 으로 추가."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _MIGRATIONS.items():
            if table not in existing_tables:
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table)}
            for col_name, col_type in columns:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))


_engine = None
_SessionLocal = None


def init_db(db_path: str):
    global _engine, _SessionLocal
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{path}", echo=False)
    Base.metadata.create_all(_engine)
    _apply_migrations(_engine)
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
