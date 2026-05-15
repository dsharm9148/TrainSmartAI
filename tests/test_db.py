"""
Tests for the database layer.

These tests verify that models are importable and structurally correct.
The connection test requires a running Postgres instance.
"""


from backend.db import models
from backend.db.session import Base, check_connection


def test_models_importable():
    """All model classes are importable without errors."""
    assert models.HealthRecord is not None
    assert models.DailyFeatures is not None
    assert models.WeeklySummary is not None
    assert models.ReadinessScore is not None
    assert models.Insight is not None
    assert models.Recommendation is not None
    assert models.ClusterAssignment is not None
    assert models.ChatSession is not None
    assert models.ChatMessage is not None


def test_all_tables_registered():
    """Every model has registered its table with Base.metadata."""
    expected_tables = {
        "health_records",
        "daily_features",
        "weekly_summaries",
        "readiness_scores",
        "insights",
        "recommendations",
        "cluster_assignments",
        "chat_sessions",
        "chat_messages",
    }
    actual_tables = set(Base.metadata.tables.keys())
    assert expected_tables == actual_tables


def test_health_record_columns():
    """HealthRecord has the expected columns."""
    cols = {c.name for c in models.HealthRecord.__table__.columns}
    assert {"id", "record_type", "source_name", "value", "unit", "start_date", "end_date"}.issubset(cols)


def test_daily_features_columns():
    """DailyFeatures has the expected columns."""
    cols = {c.name for c in models.DailyFeatures.__table__.columns}
    assert {"date", "steps", "resting_heart_rate", "sleep_duration_hrs", "workout_minutes"}.issubset(cols)


def test_readiness_score_has_components():
    """ReadinessScore stores each formula component separately."""
    cols = {c.name for c in models.ReadinessScore.__table__.columns}
    assert {"score", "sleep_score", "hr_score", "load_score", "consistency_score", "explanation"}.issubset(cols)


def test_db_connection():
    """Database is reachable (requires running Postgres)."""
    assert check_connection() is True
