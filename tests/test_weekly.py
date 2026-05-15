"""
Tests for backend/analytics/weekly.py — compute_weekly_summaries.
"""

from datetime import date, timedelta

import pytest

from backend.analytics.weekly import _nanmean, _sleep_consistency, compute_weekly_summaries
from backend.db.models import DailyFeatures, ReadinessScore, WeeklySummary

# A known Monday
MON1 = date(2025, 1, 6)
MON2 = date(2025, 1, 13)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _day(
    db,
    d: date,
    *,
    steps: int = 8000,
    sleep: float = 7.0,
    rhr: float = 55.0,
    workout_count: int = 0,
    workout_min: float = 0.0,
) -> DailyFeatures:
    row = DailyFeatures(
        date=d,
        steps=steps,
        sleep_duration_hrs=sleep,
        resting_heart_rate=rhr,
        workout_count=workout_count,
        workout_minutes=workout_min,
    )
    db.add(row)
    return row


def _readiness(db, d: date, score: float = 75.0) -> ReadinessScore:
    row = ReadinessScore(date=d, score=score)
    db.add(row)
    return row


# ─── Unit tests for helpers ───────────────────────────────────────────────────


class TestNanmean:
    def test_returns_mean(self):
        import pandas as pd
        s = pd.Series([4.0, 6.0, None])
        assert _nanmean(s) == pytest.approx(5.0)

    def test_all_nan_returns_none(self):
        import pandas as pd
        assert _nanmean(pd.Series([None, None])) is None


class TestSleepConsistency:
    def test_zero_std_gives_100(self):
        import pandas as pd
        s = pd.Series([7.0, 7.0, 7.0])
        assert _sleep_consistency(s) == pytest.approx(100.0)

    def test_high_variance_gives_low_score(self):
        import pandas as pd
        s = pd.Series([5.0, 9.0])  # std ≈ 2.83 → score ≈ 29
        score = _sleep_consistency(s)
        assert score is not None and score < 50

    def test_single_value_returns_none(self):
        import pandas as pd
        assert _sleep_consistency(pd.Series([7.0])) is None

    def test_all_nan_returns_none(self):
        import pandas as pd
        assert _sleep_consistency(pd.Series([None, None])) is None

    def test_score_clamped_at_zero(self):
        import pandas as pd
        pd.Series([4.0, 10.0])  # std = 3 → 100 - 75 = 25, still > 0
        s2 = pd.Series([1.0, 15.0])  # std >> 4 → clamped to 0
        score = _sleep_consistency(s2)
        assert score == pytest.approx(0.0)


# ─── Integration tests ────────────────────────────────────────────────────────


class TestComputeWeeklySummaries:
    def test_empty_db_returns_zero(self, db_session):
        assert compute_weekly_summaries(db_session) == 0

    def test_single_week_returns_one(self, db_session):
        for i in range(5):
            _day(db_session, MON1 + timedelta(days=i))
        db_session.flush()
        assert compute_weekly_summaries(db_session) == 1

    def test_two_weeks_returns_two(self, db_session):
        for i in range(7):
            _day(db_session, MON1 + timedelta(days=i))
        for i in range(7):
            _day(db_session, MON2 + timedelta(days=i))
        db_session.flush()
        assert compute_weekly_summaries(db_session) == 2

    def test_week_start_is_monday(self, db_session):
        # Insert a Wednesday
        _day(db_session, MON1 + timedelta(days=2))
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.week_start == MON1
        assert row.week_end == MON1 + timedelta(days=6)

    def test_avg_daily_steps(self, db_session):
        _day(db_session, MON1, steps=6000)
        _day(db_session, MON1 + timedelta(days=1), steps=10000)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.avg_daily_steps == pytest.approx(8000.0)

    def test_avg_sleep_hrs(self, db_session):
        _day(db_session, MON1, sleep=6.0)
        _day(db_session, MON1 + timedelta(days=1), sleep=8.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.avg_sleep_hrs == pytest.approx(7.0)

    def test_sleep_consistency_perfect_when_same_duration(self, db_session):
        for i in range(5):
            _day(db_session, MON1 + timedelta(days=i), sleep=7.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.sleep_consistency_score == pytest.approx(100.0)

    def test_sleep_consistency_none_for_single_day(self, db_session):
        _day(db_session, MON1, sleep=7.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.sleep_consistency_score is None

    def test_total_workout_minutes(self, db_session):
        _day(db_session, MON1, workout_count=1, workout_min=45.0)
        _day(db_session, MON1 + timedelta(days=2), workout_count=1, workout_min=60.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.total_workout_minutes == pytest.approx(105.0)

    def test_workout_days_counted_correctly(self, db_session):
        _day(db_session, MON1, workout_count=1, workout_min=45.0)
        _day(db_session, MON1 + timedelta(days=1), workout_count=0)
        _day(db_session, MON1 + timedelta(days=2), workout_count=2, workout_min=60.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.workout_days == 2

    def test_avg_readiness_score(self, db_session):
        for i in range(3):
            _day(db_session, MON1 + timedelta(days=i))
            _readiness(db_session, MON1 + timedelta(days=i), score=float(70 + i * 5))
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        # (70 + 75 + 80) / 3 = 75
        assert row.avg_readiness_score == pytest.approx(75.0)

    def test_avg_readiness_none_when_no_scores(self, db_session):
        for i in range(3):
            _day(db_session, MON1 + timedelta(days=i))
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.avg_readiness_score is None

    def test_no_sleep_data_yields_none_avg_and_consistency(self, db_session):
        db_session.add(DailyFeatures(date=MON1, steps=8000))
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.avg_sleep_hrs is None
        assert row.sleep_consistency_score is None

    def test_idempotent(self, db_session):
        for i in range(5):
            _day(db_session, MON1 + timedelta(days=i))
        db_session.flush()
        c1 = compute_weekly_summaries(db_session)
        c2 = compute_weekly_summaries(db_session)
        assert c1 == c2 == 1
        assert db_session.query(WeeklySummary).count() == 1

    def test_days_spanning_two_weeks_split_correctly(self, db_session):
        for i in range(3):
            _day(db_session, MON1 + timedelta(days=i), steps=5000)
        for i in range(3):
            _day(db_session, MON2 + timedelta(days=i), steps=10000)
        db_session.flush()
        count = compute_weekly_summaries(db_session)
        assert count == 2
        rows = (
            db_session.query(WeeklySummary)
            .order_by(WeeklySummary.week_start)
            .all()
        )
        assert rows[0].avg_daily_steps == pytest.approx(5000.0)
        assert rows[1].avg_daily_steps == pytest.approx(10000.0)

    def test_avg_resting_hr(self, db_session):
        _day(db_session, MON1, rhr=52.0)
        _day(db_session, MON1 + timedelta(days=1), rhr=58.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.avg_resting_hr == pytest.approx(55.0)

    def test_no_workout_week_stores_zero_days(self, db_session):
        for i in range(3):
            _day(db_session, MON1 + timedelta(days=i), workout_count=0, workout_min=0.0)
        db_session.flush()
        compute_weekly_summaries(db_session)
        row = db_session.query(WeeklySummary).first()
        assert row.workout_days == 0
        assert row.total_workout_minutes is None
