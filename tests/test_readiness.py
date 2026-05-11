"""
Tests for the readiness score formula (Day 8).

Fixture data (from sample_export.xml):
  Jan 1: rhr=58,  sleep=None, workouts=None
  Jan 2: rhr=60,  sleep=8.0h, workout=30.5 min
  Jan 3: rhr=57,  sleep=7.0h, workouts=None
  Jan 4: rhr=63,  sleep=8.0h, workout=45.2 min
  Jan 5: rhr=59,  sleep=7.5h, workouts=None

Rolling windows (from daily_features, past only):
  Jan 5 hr baseline  = mean(58, 60, 57, 63) = 59.5  → deviation = -0.5 → score 100
  Jan 4 hr baseline  = mean(58, 60, 57)     = 58.33 → deviation = +4.67 → score ≈ 44
  Jan 2/3 hr         < 3 prior points → absolute scale used
  Jan 5 load         = 30.5 + 45.2 = 75.7 min prior 7 days → score ≈ 83.5
  Jan 5 consistency  = std of sleep_start hours Jan 2–4 ≥ 3 points → computed
"""

from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from backend.db.models import DailyFeatures, ReadinessScore
from backend.ingestion.pipeline import run_ingestion
from backend.preprocessing.daily_features import compute_daily_features
from backend.scoring.readiness import (
    compute_readiness_scores,
    score_consistency,
    score_hr,
    score_load,
    score_sleep,
)

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"
JAN1 = date(2024, 1, 1)
JAN2 = date(2024, 1, 2)
JAN3 = date(2024, 1, 3)
JAN4 = date(2024, 1, 4)
JAN5 = date(2024, 1, 5)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def populated_db(db_session):
    run_ingestion(FIXTURE_XML, db_session)
    yield db_session


@pytest.fixture
def readiness_rows(populated_db):
    """DB with ingestion + daily features + readiness scores; keyed by date."""
    compute_daily_features(populated_db)
    compute_readiness_scores(populated_db)
    rows = populated_db.query(ReadinessScore).order_by(ReadinessScore.date).all()
    return {r.date: r for r in rows}


# ─── score_sleep ──────────────────────────────────────────────────────────────

class TestScoreSleep:
    def test_none_returns_none(self):
        assert score_sleep(None) is None

    def test_below_4h_is_zero(self):
        assert score_sleep(3.9) == 0.0
        assert score_sleep(0.0) == 0.0

    def test_exactly_4h_is_zero(self):
        assert score_sleep(4.0) == 0.0

    def test_6h_is_50(self):
        assert score_sleep(6.0) == pytest.approx(50.0, abs=0.1)

    def test_7h_is_75(self):
        assert score_sleep(7.0) == pytest.approx(75.0, abs=0.1)

    def test_7point5h_is_87point5(self):
        assert score_sleep(7.5) == pytest.approx(87.5, abs=0.1)

    def test_8h_is_100(self):
        assert score_sleep(8.0) == 100.0

    def test_9h_is_100(self):
        assert score_sleep(9.0) == 100.0

    def test_9point5h_is_100(self):
        assert score_sleep(9.5) == 100.0

    def test_10h_slight_penalty(self):
        # 100 - (10 - 9.5) * 20 = 90
        assert score_sleep(10.0) == pytest.approx(90.0, abs=0.1)

    def test_oversleep_floor_at_70(self):
        # Floor kicks in at 11h+
        assert score_sleep(11.0) == pytest.approx(70.0, abs=0.1)
        assert score_sleep(15.0) == pytest.approx(70.0, abs=0.1)

    def test_score_in_range(self):
        for hrs in [0, 3, 4, 5, 6, 7, 7.5, 8, 9, 10, 12]:
            s = score_sleep(float(hrs))
            assert 0.0 <= s <= 100.0, f"score_sleep({hrs}) = {s} out of range"


# ─── score_hr ─────────────────────────────────────────────────────────────────

class TestScoreHr:
    def test_none_returns_none(self):
        assert score_hr(None, pd.Series([], dtype=float)) is None

    def test_absolute_scale_no_history(self):
        # < 3 history points → absolute scale
        assert score_hr(50.0, pd.Series([55.0, 58.0])) == pytest.approx(100.0)

    def test_absolute_scale_rhr_60(self):
        # At 60 bpm with no history: 100 - (60-50)*2 = 80
        assert score_hr(60.0, pd.Series([], dtype=float)) == pytest.approx(80.0, abs=0.1)

    def test_absolute_scale_rhr_75(self):
        # 80 - (75-60)*2 = 50
        assert score_hr(75.0, pd.Series([], dtype=float)) == pytest.approx(50.0, abs=0.1)

    def test_relative_on_baseline_is_100(self):
        baseline = pd.Series([60.0, 60.0, 60.0])
        assert score_hr(60.0, baseline) == pytest.approx(100.0)

    def test_relative_elevated_reduces_score(self):
        # deviation = +5 → score = 100 - 60 = 40
        baseline = pd.Series([58.0, 60.0, 57.0])  # mean ≈ 58.33
        result = score_hr(63.33, baseline)         # deviation ≈ +5
        assert result == pytest.approx(40.0, abs=1.0)

    def test_relative_below_baseline_is_capped_at_100(self):
        baseline = pd.Series([60.0, 62.0, 61.0])
        result = score_hr(55.0, baseline)  # deviation = -6 → would be 172 → capped
        assert result == 100.0

    def test_relative_score_never_negative(self):
        baseline = pd.Series([50.0, 52.0, 51.0])
        result = score_hr(100.0, baseline)  # extreme elevation
        assert result >= 0.0

    def test_score_in_range(self):
        for rhr in [40, 50, 60, 70, 80, 90, 100]:
            s = score_hr(float(rhr), pd.Series([], dtype=float))
            assert 0.0 <= s <= 100.0, f"score_hr({rhr}) = {s} out of range"


# ─── score_load ───────────────────────────────────────────────────────────────

class TestScoreLoad:
    def test_zero_minutes_is_60(self):
        assert score_load(0.0) == pytest.approx(60.0)

    def test_30_minutes_between_60_and_80(self):
        s = score_load(30.0)
        assert 60.0 < s < 80.0

    def test_60_minutes_is_80(self):
        assert score_load(60.0) == pytest.approx(80.0, abs=0.1)

    def test_150_minutes_is_100(self):
        assert score_load(150.0) == pytest.approx(100.0, abs=0.1)

    def test_240_minutes_is_70(self):
        assert score_load(240.0) == pytest.approx(70.0, abs=0.1)

    def test_high_load_above_240_drops_toward_40(self):
        s = score_load(300.0)
        assert 40.0 <= s < 70.0

    def test_floor_at_40(self):
        assert score_load(1000.0) >= 40.0

    def test_score_in_range(self):
        for mins in [0, 30, 60, 100, 150, 200, 250, 300, 500]:
            s = score_load(float(mins))
            assert 0.0 <= s <= 100.0, f"score_load({mins}) = {s} out of range"


# ─── score_consistency ────────────────────────────────────────────────────────

class TestScoreConsistency:
    def _make_starts(self, hours: list[float]) -> pd.Series:
        TZ = timezone(timedelta(hours=-5))
        base = datetime(2024, 1, 1, 0, 0, tzinfo=TZ)
        return pd.Series([
            base.replace(hour=int(h), minute=int((h % 1) * 60))
            for h in hours
        ])

    def test_fewer_than_3_returns_none(self):
        assert score_consistency(pd.Series([], dtype=object)) is None
        assert score_consistency(self._make_starts([22.0, 23.0])) is None

    def test_perfect_consistency_is_100(self):
        s = score_consistency(self._make_starts([22.0, 22.0, 22.0]))
        assert s == pytest.approx(100.0)

    def test_high_variance_is_low(self):
        # std ≈ 1.76h → score ≈ 38
        s = score_consistency(self._make_starts([20.0, 22.0, 23.5]))
        assert s < 50.0

    def test_floor_at_zero(self):
        s = score_consistency(self._make_starts([20.0, 23.0, 2.0, 5.0]))
        assert s >= 0.0

    def test_score_in_range(self):
        s = score_consistency(self._make_starts([22.0, 22.5, 23.0]))
        assert 0.0 <= s <= 100.0


# ─── Integration: compute_readiness_scores ────────────────────────────────────

class TestComputeReadiness:
    def test_returns_count(self, populated_db):
        compute_daily_features(populated_db)
        count = compute_readiness_scores(populated_db)
        assert count == 5  # Jan 1–5

    def test_all_scores_in_range(self, readiness_rows):
        for d, r in readiness_rows.items():
            assert 0.0 <= r.score <= 100.0, f"score out of range on {d}: {r.score}"

    def test_component_scores_in_range_or_none(self, readiness_rows):
        for d, r in readiness_rows.items():
            for attr in ("sleep_score", "hr_score", "load_score", "consistency_score"):
                val = getattr(r, attr)
                if val is not None:
                    assert 0.0 <= val <= 100.0, f"{attr} out of range on {d}: {val}"

    def test_jan1_sleep_score_is_none(self, readiness_rows):
        """Jan 1 has no sleep data — sleep_score must be None."""
        assert readiness_rows[JAN1].sleep_score is None

    def test_jan2_sleep_score_is_100(self, readiness_rows):
        """8 hours sleep → perfect sleep score."""
        assert readiness_rows[JAN2].sleep_score == pytest.approx(100.0)

    def test_jan3_sleep_score_is_75(self, readiness_rows):
        """7 hours sleep → (7-4)/4*100 = 75."""
        assert readiness_rows[JAN3].sleep_score == pytest.approx(75.0, abs=0.1)

    def test_jan5_hr_score_is_100(self, readiness_rows):
        """Jan 5 RHR=59 is below 4-day baseline of 59.5 → score capped at 100."""
        assert readiness_rows[JAN5].hr_score == pytest.approx(100.0)

    def test_jan4_hr_score_is_elevated_rhr(self, readiness_rows):
        """Jan 4 RHR=63 is 4.67 bpm above baseline of 58.33 → score ≈ 44."""
        assert readiness_rows[JAN4].hr_score == pytest.approx(44.0, abs=2.0)

    def test_jan1_load_score_is_60(self, readiness_rows):
        """No prior workout history → resting load score of 60."""
        assert readiness_rows[JAN1].load_score == pytest.approx(60.0)

    def test_jan5_load_score_reflects_two_workouts(self, readiness_rows):
        """30.5 + 45.2 = 75.7 prior workout minutes → score ~83.5."""
        s = readiness_rows[JAN5].load_score
        assert 80.0 <= s <= 90.0

    def test_early_days_consistency_is_none(self, readiness_rows):
        """Not enough sleep_start history on Jan 1–4 to score consistency."""
        # Jan 1–4 each have fewer than 3 prior sleep_starts in window
        for d in (JAN1, JAN2, JAN3, JAN4):
            assert readiness_rows[d].consistency_score is None

    def test_jan5_has_consistency_score(self, readiness_rows):
        """Jan 5 has 3 prior sleep_start values (Jan 2, 3, 4) → scored."""
        assert readiness_rows[JAN5].consistency_score is not None

    def test_explanation_is_string(self, readiness_rows):
        for r in readiness_rows.values():
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0

    def test_idempotent(self, populated_db):
        compute_daily_features(populated_db)
        compute_readiness_scores(populated_db)
        first = {r.date: r.score for r in populated_db.query(ReadinessScore).all()}
        compute_readiness_scores(populated_db)
        second = {r.date: r.score for r in populated_db.query(ReadinessScore).all()}
        assert first == second
