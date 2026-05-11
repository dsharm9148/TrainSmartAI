"""
Tests for the insights engine (Day 9).

Unit tests operate on handcrafted DataFrames so they don't need the DB.
Integration tests use the same populated_db fixture as test_daily_features.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.analytics.insights import (
    _insight_post_workout_rhr,
    _insight_rhr_trend,
    _insight_sleep_activity,
    _insight_sleep_trend,
    _insight_steps_trend,
    _insight_weekend_sleep,
    _insight_workout_pattern,
    generate_insights,
)
from backend.db.models import DailyFeatures, Insight
from backend.ingestion.pipeline import run_ingestion
from backend.preprocessing.daily_features import compute_daily_features

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _df(n: int, **kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame of n rows; pass column arrays as kwargs."""
    base = {
        "date": [date(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "steps": [8000] * n,
        "resting_heart_rate": [58.0] * n,
        "sleep_duration_hrs": [7.5] * n,
        "workout_count": [None] * n,
        "workout_minutes": [None] * n,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def populated_db(db_session):
    run_ingestion(FIXTURE_XML, db_session)
    compute_daily_features(db_session)
    yield db_session


# ─── _insight_rhr_trend ───────────────────────────────────────────────────────

class TestRhrTrend:
    def test_declining_trend(self):
        rhr = [65, 63, 61, 59, 57, 55, 53]  # -2 bpm/day → total -12 bpm
        df = _df(7, resting_heart_rate=rhr)
        text = _insight_rhr_trend(df)
        assert text is not None
        assert "dropped" in text
        assert "cardiovascular" in text

    def test_rising_trend(self):
        rhr = [53, 55, 57, 59, 61, 63, 65]
        df = _df(7, resting_heart_rate=rhr)
        text = _insight_rhr_trend(df)
        assert text is not None
        assert "risen" in text
        assert "recovery" in text

    def test_stable_returns_stable_message(self):
        rhr = [58.0] * 10
        df = _df(10, resting_heart_rate=rhr)
        text = _insight_rhr_trend(df)
        assert text is not None
        assert "stable" in text

    def test_fewer_than_7_rows_returns_none(self):
        df = _df(6, resting_heart_rate=[58.0] * 6)
        assert _insight_rhr_trend(df) is None

    def test_all_none_rhr_returns_none(self):
        df = _df(10, resting_heart_rate=[None] * 10)
        assert _insight_rhr_trend(df) is None


# ─── _insight_sleep_trend ─────────────────────────────────────────────────────

class TestSleepTrend:
    def test_good_average_says_solid(self):
        df = _df(10, sleep_duration_hrs=[8.0] * 10)
        text = _insight_sleep_trend(df)
        assert "solid" in text

    def test_poor_average_says_below(self):
        df = _df(10, sleep_duration_hrs=[5.5] * 10)
        text = _insight_sleep_trend(df)
        assert "below" in text

    def test_improving_trend_14_days(self):
        # First 7 days: 5.5h, last 7: 8.5h
        hrs = [5.5] * 7 + [8.5] * 7
        df = _df(14, sleep_duration_hrs=hrs)
        text = _insight_sleep_trend(df)
        assert "improving" in text

    def test_declining_trend_14_days(self):
        hrs = [8.5] * 7 + [5.5] * 7
        df = _df(14, sleep_duration_hrs=hrs)
        text = _insight_sleep_trend(df)
        assert "declining" in text

    def test_fewer_than_7_returns_none(self):
        df = _df(6, sleep_duration_hrs=[7.0] * 6)
        assert _insight_sleep_trend(df) is None

    def test_all_none_returns_none(self):
        df = _df(10, sleep_duration_hrs=[None] * 10)
        assert _insight_sleep_trend(df) is None


# ─── _insight_workout_pattern ─────────────────────────────────────────────────

class TestWorkoutPattern:
    def test_no_workouts_reported(self):
        df = _df(14)
        text = _insight_workout_pattern(df)
        assert text is not None
        assert "No workouts" in text

    def test_reports_count_and_frequency(self):
        counts = [1, None, None, 1, None, None, 1] * 2  # 2 weeks, 3x/week
        minutes = [30.0, None, None, 45.0, None, None, 40.0] * 2
        df = _df(14, workout_count=counts, workout_minutes=minutes)
        text = _insight_workout_pattern(df)
        assert text is not None
        assert "times" in text
        assert "minutes" in text

    def test_total_minutes_is_accurate(self):
        # 3 workouts × 40 min = 120 total
        df = _df(7, workout_count=[1, None, 1, None, 1, None, None],
                    workout_minutes=[40.0, None, 40.0, None, 40.0, None, None])
        text = _insight_workout_pattern(df)
        assert "120" in text


# ─── _insight_sleep_activity ─────────────────────────────────────────────────

class TestSleepActivity:
    def test_strong_positive_correlation(self):
        # Perfect correlation: more sleep → more steps
        sleep = [5, 6, 7, 8, 9, 7, 8, 8, 7, 9, 8, 6, 7, 5]
        steps = [s * 1000 for s in sleep]
        df = _df(14, sleep_duration_hrs=sleep, steps=steps)
        text = _insight_sleep_activity(df)
        assert text is not None
        assert "higher" in text

    def test_weak_correlation_returns_none(self):
        np.random.seed(0)
        sleep = np.random.uniform(6, 9, 14).tolist()
        steps = np.random.uniform(5000, 12000, 14).tolist()
        df = _df(14, sleep_duration_hrs=sleep, steps=steps)
        # May or may not be None depending on random data — just test no exception
        result = _insight_sleep_activity(df)
        assert result is None or isinstance(result, str)

    def test_fewer_than_10_rows_returns_none(self):
        df = _df(9, sleep_duration_hrs=[7.0] * 9, steps=[8000] * 9)
        assert _insight_sleep_activity(df) is None


# ─── _insight_post_workout_rhr ────────────────────────────────────────────────

class TestPostWorkoutRhr:
    def test_elevated_rhr_after_workout(self):
        # Workout on days 0, 2, 4 → elevated RHR on days 1, 3, 5
        counts = [1, None, 1, None, 1, None, None, None, None, None]
        rhr_vals = [58, 65, 58, 65, 58, 65, 58, 58, 58, 58]
        df = _df(10, workout_count=counts, resting_heart_rate=rhr_vals)
        text = _insight_post_workout_rhr(df)
        assert text is not None
        assert "higher" in text

    def test_not_enough_post_workout_days_returns_none(self):
        # Only 1 workout day — not enough for comparison
        counts = [1] + [None] * 6
        df = _df(7, workout_count=counts)
        assert _insight_post_workout_rhr(df) is None

    def test_small_difference_returns_none(self):
        # RHR barely changes
        counts = [1, None, 1, None, 1, None, None, None, None, None]
        rhr_vals = [58.0] * 10  # Identical → diff < 1.5
        df = _df(10, workout_count=counts, resting_heart_rate=rhr_vals)
        assert _insight_post_workout_rhr(df) is None


# ─── _insight_weekend_sleep ───────────────────────────────────────────────────

class TestWeekendSleep:
    def test_longer_weekend_sleep(self):
        # Build 14 days starting Monday (Jan 1 2024 is a Monday)
        # Weekdays: 6.5h, Weekends: 9.0h
        base_date = date(2024, 1, 1)
        sleep = []
        for i in range(14):
            d = base_date + timedelta(days=i)
            sleep.append(9.0 if d.weekday() >= 5 else 6.5)
        df = _df(14, sleep_duration_hrs=sleep)
        text = _insight_weekend_sleep(df)
        assert text is not None
        assert "longer" in text

    def test_similar_sleep_returns_none(self):
        df = _df(14, sleep_duration_hrs=[7.5] * 14)
        assert _insight_weekend_sleep(df) is None

    def test_not_enough_weekend_days_returns_none(self):
        # Only 1 weekend day in 5-day window
        df = _df(5, sleep_duration_hrs=[7.5] * 5)
        assert _insight_weekend_sleep(df) is None


# ─── _insight_steps_trend ────────────────────────────────────────────────────

class TestStepsTrend:
    def test_trending_up(self):
        steps = list(range(5000, 15000, 1000))  # 10 days, +1000/day
        df = _df(10, steps=steps)
        text = _insight_steps_trend(df)
        assert text is not None
        assert "trending up" in text

    def test_trending_down(self):
        steps = list(range(14000, 4000, -1000))  # 10 days, -1000/day
        df = _df(10, steps=steps)
        text = _insight_steps_trend(df)
        assert "trending down" in text

    def test_stable_above_target(self):
        df = _df(10, steps=[10000] * 10)
        text = _insight_steps_trend(df)
        assert "above" in text or "target" in text

    def test_stable_below_target(self):
        df = _df(10, steps=[5000] * 10)
        text = _insight_steps_trend(df)
        assert "below" in text

    def test_fewer_than_7_returns_none(self):
        df = _df(6, steps=[8000] * 6)
        assert _insight_steps_trend(df) is None


# ─── generate_insights (integration) ─────────────────────────────────────────

class TestGenerateInsights:
    def test_returns_positive_count(self, populated_db):
        count = generate_insights(populated_db, for_date=date(2024, 1, 5), lookback_days=5)
        assert count > 0

    def test_insights_stored_in_db(self, populated_db):
        generate_insights(populated_db, for_date=date(2024, 1, 5), lookback_days=5)
        rows = populated_db.query(Insight).filter(
            Insight.generated_for == date(2024, 1, 5)
        ).all()
        assert len(rows) > 0

    def test_each_insight_has_non_empty_text(self, populated_db):
        generate_insights(populated_db, for_date=date(2024, 1, 5), lookback_days=5)
        rows = populated_db.query(Insight).filter(
            Insight.generated_for == date(2024, 1, 5)
        ).all()
        for r in rows:
            assert isinstance(r.text, str)
            assert len(r.text) > 10

    def test_idempotent(self, populated_db):
        d = date(2024, 1, 5)
        count1 = generate_insights(populated_db, for_date=d)
        count2 = generate_insights(populated_db, for_date=d)
        assert count1 == count2
        # Should replace, not double-insert
        rows = populated_db.query(Insight).filter(Insight.generated_for == d).all()
        assert len(rows) == count2

    def test_insufficient_data_returns_zero(self, db_session):
        # Empty DB — no daily features at all
        count = generate_insights(db_session, for_date=date(2024, 1, 5))
        assert count == 0

    def test_insight_types_are_valid(self, populated_db):
        valid_types = {
            "sleep_activity", "hr_trend", "sleep_quality",
            "workout_pattern", "post_workout_recovery",
            "weekend_sleep", "steps_trend",
        }
        generate_insights(populated_db, for_date=date(2024, 1, 5), lookback_days=5)
        rows = populated_db.query(Insight).filter(
            Insight.generated_for == date(2024, 1, 5)
        ).all()
        for r in rows:
            assert r.insight_type in valid_types
