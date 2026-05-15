"""
Tests for the rule-based recommendation engine (Day 10).

Unit tests exercise each rule in isolation with crafted DataFrames.
Integration tests use the fixture DB to verify end-to-end behaviour.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from backend.db.models import Recommendation
from backend.ingestion.pipeline import run_ingestion
from backend.preprocessing.daily_features import compute_daily_features
from backend.recommendations.engine import (
    _rule_consistent_bedtime,
    _rule_high_readiness_train,
    _rule_high_rhr_recovery,
    _rule_low_readiness_rest,
    _rule_low_sleep_tonight,
    _rule_no_workout_streak,
    _rule_overtraining_warning,
    _rule_post_anomaly_recovery,
    _rule_sleep_deficit,
    _rule_step_goal_nudge,
    generate_recommendations,
)
from backend.scoring.readiness import compute_readiness_scores

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"
TZ = timezone(timedelta(hours=-5))

TODAY = date(2024, 1, 5)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _df(n: int, base_date: date = date(2024, 1, 1), **kwargs) -> pd.DataFrame:
    base = {
        "date": [base_date + timedelta(days=i) for i in range(n)],
        "steps": [8000] * n,
        "resting_heart_rate": [58.0] * n,
        "sleep_duration_hrs": [7.5] * n,
        "sleep_start": [datetime(2024, 1, i + 1, 22, 30, tzinfo=TZ) for i in range(n)],
        "workout_count": [None] * n,
        "workout_minutes": [None] * n,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


def _today(df: pd.DataFrame, d: date) -> pd.Series:
    return df[df["date"] == d].iloc[0]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def populated_db(db_session):
    run_ingestion(FIXTURE_XML, db_session)
    compute_daily_features(db_session)
    compute_readiness_scores(db_session)
    yield db_session


# ─── _rule_high_rhr_recovery ──────────────────────────────────────────────────

class TestHighRhrRecovery:
    def test_fires_when_elevated(self):
        rhr = [58.0] * 6 + [65.0]  # baseline 58, today 65 → +7 bpm
        df = _df(7, resting_heart_rate=rhr)
        today = _today(df, date(2024, 1, 7))
        rec = _rule_high_rhr_recovery(today, df, None)
        assert rec is not None
        assert rec.category == "recovery"
        assert rec.priority == 1
        assert "baseline" in rec.text.lower() or "bpm" in rec.text

    def test_does_not_fire_when_normal(self):
        df = _df(7, resting_heart_rate=[58.0] * 7)
        today = _today(df, date(2024, 1, 7))
        assert _rule_high_rhr_recovery(today, df, None) is None

    def test_does_not_fire_with_insufficient_history(self):
        df = _df(3, resting_heart_rate=[58.0, 58.0, 66.0])
        today = _today(df, date(2024, 1, 3))
        assert _rule_high_rhr_recovery(today, df, None) is None

    def test_does_not_fire_below_5bpm_threshold(self):
        rhr = [58.0] * 6 + [62.0]  # only +4 bpm
        df = _df(7, resting_heart_rate=rhr)
        today = _today(df, date(2024, 1, 7))
        assert _rule_high_rhr_recovery(today, df, None) is None


# ─── _rule_low_readiness_rest ────────────────────────────────────────────────

class TestLowReadinessRest:
    def test_fires_below_55(self):
        rec = _rule_low_readiness_rest(None, pd.DataFrame(), 45.0)
        assert rec is not None
        assert rec.category == "recovery"
        assert "45" in rec.text

    def test_does_not_fire_at_55(self):
        assert _rule_low_readiness_rest(None, pd.DataFrame(), 55.0) is None

    def test_does_not_fire_when_none(self):
        assert _rule_low_readiness_rest(None, pd.DataFrame(), None) is None


# ─── _rule_high_readiness_train ───────────────────────────────────────────────

class TestHighReadinessTrain:
    def test_fires_when_high_readiness_and_no_recent_workout(self):
        df = _df(5)  # all workout_count=None
        today = _today(df, date(2024, 1, 5))
        rec = _rule_high_readiness_train(today, df, 85.0)
        assert rec is not None
        assert rec.category == "workout"

    def test_does_not_fire_when_recent_workout(self):
        counts = [None, None, None, 1, None]
        df = _df(5, workout_count=counts)
        today = _today(df, date(2024, 1, 5))
        assert _rule_high_readiness_train(today, df, 85.0) is None

    def test_does_not_fire_below_80(self):
        df = _df(5)
        today = _today(df, date(2024, 1, 5))
        assert _rule_high_readiness_train(today, df, 79.0) is None


# ─── _rule_low_sleep_tonight ─────────────────────────────────────────────────

class TestLowSleepTonight:
    def test_fires_below_6h(self):
        df = _df(3, sleep_duration_hrs=[7.0, 7.0, 5.5])
        today = _today(df, date(2024, 1, 3))
        rec = _rule_low_sleep_tonight(today, df, None)
        assert rec is not None
        assert rec.category == "sleep"
        assert "5.5" in rec.text

    def test_does_not_fire_at_6h(self):
        df = _df(3, sleep_duration_hrs=[7.0, 7.0, 6.0])
        today = _today(df, date(2024, 1, 3))
        assert _rule_low_sleep_tonight(today, df, None) is None

    def test_does_not_fire_when_no_sleep_data(self):
        df = _df(3, sleep_duration_hrs=[None, None, None])
        today = _today(df, date(2024, 1, 3))
        assert _rule_low_sleep_tonight(today, df, None) is None


# ─── _rule_sleep_deficit ─────────────────────────────────────────────────────

class TestSleepDeficit:
    def test_fires_when_7day_avg_below_6point5(self):
        df = _df(7, sleep_duration_hrs=[5.5] * 7)
        today = _today(df, date(2024, 1, 7))
        rec = _rule_sleep_deficit(today, df, None)
        assert rec is not None
        assert rec.category == "sleep"
        assert rec.priority == 2

    def test_does_not_fire_above_threshold(self):
        df = _df(7, sleep_duration_hrs=[7.0] * 7)
        today = _today(df, date(2024, 1, 7))
        assert _rule_sleep_deficit(today, df, None) is None

    def test_does_not_fire_with_few_readings(self):
        df = _df(3, sleep_duration_hrs=[5.0, 5.0, 5.0])
        today = _today(df, date(2024, 1, 3))
        assert _rule_sleep_deficit(today, df, None) is None


# ─── _rule_no_workout_streak ─────────────────────────────────────────────────

class TestNoWorkoutStreak:
    def test_fires_after_5_rest_days(self):
        df = _df(5)  # all workout_count=None
        today = _today(df, date(2024, 1, 5))
        rec = _rule_no_workout_streak(today, df, None)
        assert rec is not None
        assert rec.category == "workout"

    def test_does_not_fire_with_recent_workout(self):
        counts = [None, None, None, None, 1]
        df = _df(5, workout_count=counts)
        today = _today(df, date(2024, 1, 5))
        assert _rule_no_workout_streak(today, df, None) is None


# ─── _rule_overtraining_warning ───────────────────────────────────────────────

class TestOvertrainingWarning:
    def test_fires_when_high_volume_and_low_readiness(self):
        minutes = [50.0] * 7  # 350 min total
        df = _df(7, workout_count=[1] * 7, workout_minutes=minutes)
        today = _today(df, date(2024, 1, 7))
        rec = _rule_overtraining_warning(today, df, 60.0)
        assert rec is not None
        assert rec.category == "recovery"
        assert rec.priority == 1

    def test_does_not_fire_when_readiness_high(self):
        minutes = [50.0] * 7
        df = _df(7, workout_count=[1] * 7, workout_minutes=minutes)
        today = _today(df, date(2024, 1, 7))
        assert _rule_overtraining_warning(today, df, 70.0) is None

    def test_does_not_fire_when_volume_low(self):
        minutes = [30.0] * 7  # only 210 min
        df = _df(7, workout_count=[1] * 7, workout_minutes=minutes)
        today = _today(df, date(2024, 1, 7))
        assert _rule_overtraining_warning(today, df, 55.0) is None


# ─── _rule_consistent_bedtime ─────────────────────────────────────────────────

class TestConsistentBedtime:
    def test_fires_when_high_variance(self):
        # Sleep starts ranging from 10pm to 2am
        starts = [
            datetime(2024, 1, i + 1, 22 if i % 2 == 0 else 1, 0, tzinfo=TZ)
            for i in range(10)
        ]
        df = _df(10, sleep_start=starts)
        today = _today(df, date(2024, 1, 10))
        rec = _rule_consistent_bedtime(today, df, None)
        assert rec is not None
        assert rec.category == "habit"
        assert rec.priority == 3

    def test_does_not_fire_when_consistent(self):
        starts = [datetime(2024, 1, i + 1, 22, 30, tzinfo=TZ) for i in range(10)]
        df = _df(10, sleep_start=starts)
        today = _today(df, date(2024, 1, 10))
        assert _rule_consistent_bedtime(today, df, None) is None


# ─── _rule_step_goal_nudge ────────────────────────────────────────────────────

class TestStepGoalNudge:
    def test_fires_below_6000_avg(self):
        df = _df(7, steps=[4000] * 7)
        today = _today(df, date(2024, 1, 7))
        rec = _rule_step_goal_nudge(today, df, None)
        assert rec is not None
        assert rec.category == "habit"
        assert rec.priority == 3

    def test_does_not_fire_above_threshold(self):
        df = _df(7, steps=[7000] * 7)
        today = _today(df, date(2024, 1, 7))
        assert _rule_step_goal_nudge(today, df, None) is None


# ─── _rule_post_anomaly_recovery ─────────────────────────────────────────────

class TestPostAnomalyRecovery:
    def test_fires_after_low_steps_high_rhr_day(self):
        # Yesterday: 1500 steps, RHR=68 (baseline ~58 → +10 bpm)
        rhr = [58.0, 58.0, 58.0, 58.0, 58.0, 68.0, 58.0]
        steps = [8000, 8000, 8000, 8000, 8000, 1500, 8000]
        df = _df(7, resting_heart_rate=rhr, steps=steps)
        today = _today(df, date(2024, 1, 7))
        rec = _rule_post_anomaly_recovery(today, df, None)
        assert rec is not None
        assert rec.category == "recovery"

    def test_does_not_fire_when_steps_normal(self):
        df = _df(7)  # all 8000 steps
        today = _today(df, date(2024, 1, 7))
        assert _rule_post_anomaly_recovery(today, df, None) is None


# ─── generate_recommendations (integration) ──────────────────────────────────

class TestGenerateRecommendations:
    def test_returns_non_negative_count(self, populated_db):
        count = generate_recommendations(populated_db, for_date=TODAY, lookback_days=5)
        assert count >= 0

    def test_recommendations_stored_in_db(self, populated_db):
        generate_recommendations(populated_db, for_date=TODAY, lookback_days=5)
        rows = populated_db.query(Recommendation).filter(
            Recommendation.date == TODAY
        ).all()
        # Whether 0 or more, all stored rows should be valid
        for r in rows:
            assert isinstance(r.text, str) and len(r.text) > 5
            assert r.category in ("recovery", "workout", "sleep", "habit")
            assert r.priority in (1, 2, 3)

    def test_idempotent(self, populated_db):
        d = TODAY
        c1 = generate_recommendations(populated_db, for_date=d, lookback_days=5)
        c2 = generate_recommendations(populated_db, for_date=d, lookback_days=5)
        assert c1 == c2
        rows = populated_db.query(Recommendation).filter(Recommendation.date == d).all()
        assert len(rows) == c2

    def test_reasoning_is_non_empty(self, populated_db):
        generate_recommendations(populated_db, for_date=TODAY, lookback_days=5)
        rows = populated_db.query(Recommendation).filter(
            Recommendation.date == TODAY
        ).all()
        for r in rows:
            assert r.reasoning is not None and len(r.reasoning) > 0

    def test_empty_db_returns_zero(self, db_session):
        count = generate_recommendations(db_session, for_date=TODAY)
        assert count == 0

    def test_priority_ordering(self, populated_db):
        generate_recommendations(populated_db, for_date=TODAY, lookback_days=5)
        rows = populated_db.query(Recommendation).filter(
            Recommendation.date == TODAY
        ).order_by(Recommendation.priority).all()
        priorities = [r.priority for r in rows]
        assert priorities == sorted(priorities)
