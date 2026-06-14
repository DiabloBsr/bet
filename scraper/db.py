"""Database engine + scoped session helper."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from scraper.config import Settings
from scraper.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def init_engine(settings: Settings) -> Engine:
    global _engine, _SessionLocal

    if settings.db_url.startswith("sqlite"):
        db_path = settings.db_url.replace("sqlite:///", "").lstrip("/")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(settings.db_url, future=True, echo=False)
    _SessionLocal = sessionmaker(
        bind=_engine, autoflush=False, autocommit=False, future=True
    )
    Base.metadata.create_all(_engine)
    _apply_migrations(_engine)
    return _engine


def _apply_migrations(engine: Engine) -> None:
    """Petites migrations idempotentes — appelées après create_all."""
    insp = inspect(engine)
    existing_events = {c["name"] for c in insp.get_columns("events")}
    if "expected_start" not in existing_events:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE events ADD COLUMN expected_start DATETIME"))

    existing_results = {c["name"] for c in insp.get_columns("results")}
    for col, sql_type in [
        ("ht_score_a", "INTEGER"),
        ("ht_score_b", "INTEGER"),
        ("goals_json", "TEXT"),  # SQLite stocke JSON en TEXT
    ]:
        if col not in existing_results:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE results ADD COLUMN {col} {sql_type}"))


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_engine() first.")
    session: Session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
