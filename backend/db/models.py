"""
SQLAlchemy ORM models — one class per database table.

Each class maps directly to the schema in PLAN.md. Models use SQLAlchemy 2.0
style (Mapped + mapped_column) for explicit type annotations.
"""

from __future__ import annotations

import uuid as uuid_module
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from backend.db.session import Base

# Convenience alias — maps to TIMESTAMP WITH TIME ZONE in Postgres
TZ = DateTime(timezone=True)


class HealthRecord(Base):
    """Raw Apple Health data points — one row per individual record in the export.

    The composite unique constraint on (record_type, source_name, start_date, end_date)
    makes re-ingestion safe: duplicate records are silently skipped on upsert.
    """

    __tablename__ = "health_records"
    __table_args__ = (
        UniqueConstraint(
            "record_type", "source_name", "start_date", "end_date",
            name="uq_health_record",
        ),
        Index("idx_health_records_type_date", "record_type", "start_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    record_type: Mapped[str] = mapped_column(String(120), nullable=False)
    source_name: Mapped[Optional[str]] = mapped_column(String(100))
    value: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String(50))
    start_date: Mapped[datetime] = mapped_column(TZ, nullable=False)
    end_date: Mapped[datetime] = mapped_column(TZ, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())


class DailyFeatures(Base):
    """Aggregated daily metrics — one row per calendar day.

    Built by the preprocessing layer from health_records. Stores None (not 0)
    for metrics with no data that day, so missing data stays distinguishable.
    """

    __tablename__ = "daily_features"
    __table_args__ = (Index("idx_daily_features_date", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    steps: Mapped[Optional[int]] = mapped_column(Integer)
    avg_heart_rate: Mapped[Optional[float]] = mapped_column(Float)
    resting_heart_rate: Mapped[Optional[float]] = mapped_column(Float)
    sleep_duration_hrs: Mapped[Optional[float]] = mapped_column(Float)
    sleep_start: Mapped[Optional[datetime]] = mapped_column(TZ)
    sleep_end: Mapped[Optional[datetime]] = mapped_column(TZ)
    workout_count: Mapped[Optional[int]] = mapped_column(Integer)
    workout_minutes: Mapped[Optional[float]] = mapped_column(Float)
    workout_calories: Mapped[Optional[float]] = mapped_column(Float)
    active_energy_kcal: Mapped[Optional[float]] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        TZ, server_default=func.now(), onupdate=func.now()
    )


class WeeklySummary(Base):
    """Weekly rollups — one row per Monday–Sunday week."""

    __tablename__ = "weekly_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    week_start: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    week_end: Mapped[date] = mapped_column(Date, nullable=False)
    avg_daily_steps: Mapped[Optional[float]] = mapped_column(Float)
    avg_sleep_hrs: Mapped[Optional[float]] = mapped_column(Float)
    # Lower std dev = more consistent sleep schedule
    sleep_consistency_score: Mapped[Optional[float]] = mapped_column(Float)
    avg_resting_hr: Mapped[Optional[float]] = mapped_column(Float)
    total_workout_minutes: Mapped[Optional[float]] = mapped_column(Float)
    workout_days: Mapped[Optional[int]] = mapped_column(Integer)
    avg_readiness_score: Mapped[Optional[float]] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        TZ, server_default=func.now(), onupdate=func.now()
    )


class ReadinessScore(Base):
    """Daily readiness score with per-component breakdown.

    Storing each component separately (sleep, HR, load, consistency) makes the
    formula transparent and easy to explain — both in the UI and in interviews.
    Score range: 0–100.
    """

    __tablename__ = "readiness_scores"
    __table_args__ = (Index("idx_readiness_date", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    sleep_score: Mapped[Optional[float]] = mapped_column(Float)
    hr_score: Mapped[Optional[float]] = mapped_column(Float)
    load_score: Mapped[Optional[float]] = mapped_column(Float)
    consistency_score: Mapped[Optional[float]] = mapped_column(Float)
    explanation: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        TZ, server_default=func.now(), onupdate=func.now()
    )


class Insight(Base):
    """Plain-English observations generated from correlation analysis."""

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_for: Mapped[date] = mapped_column(Date, nullable=False)
    # e.g. 'sleep_activity', 'hr_trend', 'workout_pattern', 'weekly'
    insight_type: Mapped[Optional[str]] = mapped_column(String(60))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())


class Recommendation(Base):
    """Personalized rule-based suggestions with stored reasoning.

    The reasoning column records which rule fired and why, making the
    recommendation engine auditable and easy to explain.
    """

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    # 'recovery', 'workout', 'sleep', 'habit'
    category: Mapped[Optional[str]] = mapped_column(String(50))
    # 1 = high priority, 2 = medium, 3 = low
    priority: Mapped[Optional[int]] = mapped_column(SmallInteger)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())


class ClusterAssignment(Base):
    """K-means cluster label assigned to each day."""

    __tablename__ = "cluster_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, unique=True, nullable=False)
    cluster_id: Mapped[Optional[int]] = mapped_column(SmallInteger)
    # e.g. 'high sleep, active day' — set after manual cluster labeling
    cluster_label: Mapped[Optional[str]] = mapped_column(String(100))
    updated_at: Mapped[datetime] = mapped_column(
        TZ, server_default=func.now(), onupdate=func.now()
    )


class ChatSession(Base):
    """Groups a multi-turn assistant conversation."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[uuid_module.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, default=uuid_module.uuid4
    )
    started_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())
    last_active: Mapped[datetime] = mapped_column(
        TZ, server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    """Individual turns within a chat session."""

    __tablename__ = "chat_messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_message_role"),
        Index("idx_chat_messages_session", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[uuid_module.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ, server_default=func.now())
