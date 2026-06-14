"""SQLAlchemy ORM models — schéma de l'étape 4."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # match_key = "competition|round_info|team_a|team_b|YYYYMMDDHHMM" — clé unique
    # par occurrence (le timestamp évite que les rounds cycliques réutilisent un
    # vieux event et qu'on attache d'anciens résultats à de nouvelles rencontres).
    match_key: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    sport: Mapped[str | None] = mapped_column(String(64), nullable=True)
    competition: Mapped[str | None] = mapped_column(String(128), nullable=True)
    team_a: Mapped[str | None] = mapped_column(String(128), nullable=True)
    team_b: Mapped[str | None] = mapped_column(String(128), nullable=True)
    round_info: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    odds = relationship("OddsSnapshot", back_populates="event", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="event", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("match_key", name="uq_event_match_key"),)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    odds_home: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_away: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra_markets: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # SHA-256 du payload — empêche les insertions strictement identiques
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    scrape_run_id: Mapped[int | None] = mapped_column(ForeignKey("scrape_runs.id"), nullable=True)

    event = relationship("Event", back_populates="odds")


class Result(Base):
    __tablename__ = "results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), index=True
    )
    score_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_score: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Mi-temps : populates a partir du payload /results (halfTimeScore: "1:1")
    ht_score_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ht_score_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Liste des buts avec timing : [{minute, homeScore, awayScore, team}, ...]
    goals_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    scrape_run_id: Mapped[int | None] = mapped_column(ForeignKey("scrape_runs.id"), nullable=True)

    event = relationship("Event", back_populates="results")

    __table_args__ = (UniqueConstraint("event_id", name="uq_result_event"),)


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    events_seen: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    snapshots_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    results_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rankings_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class RankingSnapshot(Base):
    __tablename__ = "rankings_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    competition: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    team_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    won: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draw: Mapped[int | None] = mapped_column(Integer, nullable=True)
    history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    scrape_run_id: Mapped[int | None] = mapped_column(ForeignKey("scrape_runs.id"), nullable=True)
