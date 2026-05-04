"""
Tests for the daily features computation layer.

All tests require a running Postgres instance.
The `populated_db` fixture runs the full ingestion pipeline first,
then tests verify the aggregated daily_features rows.

Expected fixture data (from sample_export.xml):
  Jan 1: steps=8543, avg_hr=68.3, rhr=58,  sleep=None,  workouts=None
  Jan 2: steps=10234, avg_hr=70.0, rhr=60, sleep=8.0h,  workout=30.5min running
  Jan 3: steps=6789,  avg_hr=75.0, rhr=57, sleep=7.0h,  workouts=None
  Jan 4: steps=11502, avg_hr=82.0, rhr=63, sleep=8.0h,  workout=45.2min cycling
  Jan 5: steps=7210,  avg_hr=69.0, rhr=59, sleep=7.5h,  workouts=None
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from backend.db.models import DailyFeatures
from backend.ingestion.pipeline import run_ingestion
from backend.preprocessing.daily_features import compute_daily_features

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"

JAN1 = date(2024, 1, 1)
JAN2 = date(2024, 1, 2)
JAN3 = date(2024, 1, 3)
JAN4 = date(2024, 1, 4)
JAN5 = date(2024, 1, 5)


@pytest.fixture
def populated_db(db_session):
    """DB session pre-loaded with the sample fixture data."""
    run_ingestion(FIXTURE_XML, db_session)
    yield db_session


@pytest.fixture
def daily_rows(populated_db):
    """Compute daily features and return rows keyed by date."""
    compute_daily_features(populated_db)
    rows = populated_db.query(DailyFeatures).order_by(DailyFeatures.date).all()
    return {r.date: r for r in rows}


# ─── Row count ────────────────────────────────────────────────────────────────

def test_correct_number_of_days(populated_db):
    """One row is created per day that has any health data."""
    count = compute_daily_features(populated_db)
    assert count == 5  # Jan 1–5


def test_returns_five_daily_rows(daily_rows):
    assert len(daily_rows) == 5
    assert JAN1 in daily_rows
    assert JAN5 in daily_rows


# ─── Steps ────────────────────────────────────────────────────────────────────

def test_steps_jan1(daily_rows):
    assert daily_rows[JAN1].steps == 8543

def test_steps_jan2(daily_rows):
    assert daily_rows[JAN2].steps == 10234

def test_steps_jan3(daily_rows):
    assert daily_rows[JAN3].steps == 6789

def test_steps_jan4(daily_rows):
    assert daily_rows[JAN4].steps == 11502

def test_steps_jan5(daily_rows):
    assert daily_rows[JAN5].steps == 7210


# ─── Resting heart rate ───────────────────────────────────────────────────────

def test_rhr_jan1(daily_rows):
    assert daily_rows[JAN1].resting_heart_rate == pytest.approx(58.0)

def test_rhr_jan3(daily_rows):
    assert daily_rows[JAN3].resting_heart_rate == pytest.approx(57.0)


# ─── Sleep attribution (end_date day = wake-up day) ───────────────────────────

def test_sleep_jan1_is_none(daily_rows):
    """Jan 1 has no sleep — last night's sleep ends on Jan 2."""
    assert daily_rows[JAN1].sleep_duration_hrs is None

def test_sleep_jan2_is_8_hours(daily_rows):
    """Sleep from Jan 1 23:00 → Jan 2 07:00 = 8 hours, attributed to Jan 2."""
    assert daily_rows[JAN2].sleep_duration_hrs == pytest.approx(8.0, rel=0.01)

def test_sleep_jan3_is_7_hours(daily_rows):
    """Sleep from Jan 2 23:30 → Jan 3 06:30 = 7 hours."""
    assert daily_rows[JAN3].sleep_duration_hrs == pytest.approx(7.0, rel=0.01)

def test_sleep_jan4_is_8_hours(daily_rows):
    """Sleep from Jan 3 22:00 → Jan 4 06:00 = 8 hours."""
    assert daily_rows[JAN4].sleep_duration_hrs == pytest.approx(8.0, rel=0.01)

def test_sleep_jan5_is_7point5_hours(daily_rows):
    """Sleep from Jan 4 23:00 → Jan 5 06:30 = 7.5 hours."""
    assert daily_rows[JAN5].sleep_duration_hrs == pytest.approx(7.5, rel=0.01)


# ─── Workouts ─────────────────────────────────────────────────────────────────

def test_no_workout_jan1(daily_rows):
    assert daily_rows[JAN1].workout_count is None

def test_workout_jan2(daily_rows):
    assert daily_rows[JAN2].workout_count == 1
    assert daily_rows[JAN2].workout_minutes == pytest.approx(30.5, rel=0.01)

def test_no_workout_jan3(daily_rows):
    assert daily_rows[JAN3].workout_count is None

def test_workout_jan4(daily_rows):
    assert daily_rows[JAN4].workout_count == 1
    assert daily_rows[JAN4].workout_minutes == pytest.approx(45.2, rel=0.01)


# ─── Idempotency ──────────────────────────────────────────────────────────────

def test_compute_is_idempotent(populated_db):
    """Running compute twice produces the same results."""
    compute_daily_features(populated_db)
    first = {
        r.date: r.steps
        for r in populated_db.query(DailyFeatures).all()
    }

    compute_daily_features(populated_db)
    second = {
        r.date: r.steps
        for r in populated_db.query(DailyFeatures).all()
    }

    assert first == second


# ─── Missing data handling ────────────────────────────────────────────────────

def test_missing_metrics_are_none_not_zero(daily_rows):
    """Days without sleep store None, not 0."""
    assert daily_rows[JAN1].sleep_duration_hrs is None
    assert daily_rows[JAN1].workout_count is None
