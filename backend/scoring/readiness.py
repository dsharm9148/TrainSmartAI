"""
Readiness score formula.

Produces a 0–100 score per day from four components, each independently
interpretable. Missing components (no data) are excluded and the remaining
weights are renormalized so the score is always meaningful even with partial data.

Component weights (from config, must sum to 1.0):
  sleep        0.40  — hours slept last night
  hr           0.30  — resting HR deviation from 28-day personal baseline
  load         0.20  — 7-day training volume (too little or too much both hurt)
  consistency  0.10  — std dev of sleep-start times over the past 14 days

Public API:
  compute_readiness_scores(db, start_date, end_date) -> int
  score_sleep(sleep_hrs)                             -> Optional[float]
  score_hr(today_rhr, recent_rhr_series)             -> Optional[float]
  score_load(recent_workout_minutes)                 -> float
  score_consistency(sleep_starts_series)             -> Optional[float]
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import DailyFeatures, ReadinessScore

# ─── Public API ───────────────────────────────────────────────────────────────


def compute_readiness_scores(
    db: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> int:
    """
    Compute and upsert ReadinessScore rows for all days that have daily feature data.

    Loads a 30-day lookback buffer before start_date so rolling HR and consistency
    windows are accurate even at the beginning of the requested range.
    Returns the number of days scored.
    """
    lookback_start = (start_date - timedelta(days=30)) if start_date else None
    df = _load_features(db, lookback_start, end_date)

    if df.empty:
        return 0

    target_dates = _target_dates(df, start_date, end_date)
    scored = [_score_day(d, df) for d in target_dates]
    scored = [r for r in scored if r is not None]

    if scored:
        _upsert(db, scored)

    return len(scored)


def score_sleep(sleep_hrs: Optional[float]) -> Optional[float]:
    """
    Sleep score from hours slept (0–100).

    Piecewise:
      < 4h       → 0   (severely sleep deprived)
      4–8h       → linear 0→100
      8–9.5h     → 100 (optimal)
      > 9.5h     → slight penalty (oversleeping, floor 70)
    """
    if sleep_hrs is None:
        return None
    if sleep_hrs < 4.0:
        return 0.0
    if sleep_hrs < 8.0:
        return round((sleep_hrs - 4.0) / 4.0 * 100.0, 1)
    if sleep_hrs <= 9.5:
        return 100.0
    return round(max(70.0, 100.0 - (sleep_hrs - 9.5) * 20.0), 1)


def score_hr(
    today_rhr: Optional[float],
    recent_rhr: pd.Series,
) -> Optional[float]:
    """
    HR score from resting heart rate (0–100).

    Uses deviation from personal 28-day baseline when ≥ 3 prior readings exist,
    otherwise falls back to an absolute scale calibrated for adults.

    Relative formula: score = 100 − deviation × 12
      deviation = 0 bpm  → 100  (on baseline)
      deviation = +5 bpm → 40   (elevated, likely fatigued)
      deviation = −3 bpm → 100  (below baseline, well recovered)
    """
    if today_rhr is None:
        return None

    if len(recent_rhr) >= 3:
        baseline = float(recent_rhr.mean())
        deviation = today_rhr - baseline
        return round(float(max(0.0, min(100.0, 100.0 - deviation * 12.0))), 1)

    return _absolute_rhr_score(today_rhr)


def score_load(recent_workout_minutes: float) -> float:
    """
    Training load score from 7-day total workout minutes (0–100).

    Piecewise optimum at 60–150 minutes/week:
      0 min        → 60  (rested but inactive)
      0–60 min     → 60→80
      60–150 min   → 80→100  (optimal)
      150–240 min  → 100→70
      > 240 min    → 70→40   (high load, recovery needed)
    """
    if recent_workout_minutes <= 0:
        return 60.0
    if recent_workout_minutes < 60:
        return round(60.0 + (recent_workout_minutes / 60.0) * 20.0, 1)
    if recent_workout_minutes <= 150:
        return round(80.0 + ((recent_workout_minutes - 60) / 90.0) * 20.0, 1)
    if recent_workout_minutes <= 240:
        return round(100.0 - ((recent_workout_minutes - 150) / 90.0) * 30.0, 1)
    return round(max(40.0, 70.0 - ((recent_workout_minutes - 240) / 60.0) * 10.0), 1)


def score_consistency(sleep_starts: pd.Series) -> Optional[float]:
    """
    Sleep schedule consistency score from std dev of sleep-start times (0–100).

    Requires ≥ 3 data points; returns None if insufficient history.
    std_dev = 0h → 100, std_dev ≥ 3h → 0.
    """
    if len(sleep_starts) < 3:
        return None

    hours = sleep_starts.apply(lambda dt: dt.hour + dt.minute / 60.0)
    std_hours = float(hours.std())
    return round(max(0.0, 100.0 - std_hours * 35.0), 1)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _absolute_rhr_score(rhr: float) -> float:
    """Absolute RHR score used when rolling history is too short."""
    if rhr <= 50:
        return 100.0
    if rhr <= 60:
        return round(100.0 - (rhr - 50) * 2.0, 1)   # 100→80
    if rhr <= 75:
        return round(80.0 - (rhr - 60) * 2.0, 1)    # 80→50
    if rhr <= 90:
        return round(50.0 - (rhr - 75) * 2.0, 1)    # 50→20
    return round(max(0.0, 20.0 - (rhr - 90) * 2.0), 1)


def _load_features(
    db: Session,
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.DataFrame:
    q = db.query(DailyFeatures)
    if start_date:
        q = q.filter(DailyFeatures.date >= start_date)
    if end_date:
        q = q.filter(DailyFeatures.date <= end_date)

    rows = q.order_by(DailyFeatures.date).all()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "date": r.date,
            "resting_heart_rate": r.resting_heart_rate,
            "sleep_duration_hrs": r.sleep_duration_hrs,
            "sleep_start": r.sleep_start,
            "workout_minutes": r.workout_minutes,
        }
        for r in rows
    ])


def _target_dates(
    df: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
) -> list[date]:
    dates = sorted(df["date"].tolist())
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    return dates


def _score_day(d: date, df: pd.DataFrame) -> Optional[dict]:
    today_row = df[df["date"] == d]
    if today_row.empty:
        return None

    today = today_row.iloc[0]

    # Rolling lookback windows — exclude today so we're not using future data
    past_28 = df[(df["date"] >= d - timedelta(days=28)) & (df["date"] < d)]
    past_14 = df[(df["date"] >= d - timedelta(days=14)) & (df["date"] < d)]
    past_7 = df[(df["date"] >= d - timedelta(days=7)) & (df["date"] < d)]

    sleep_s = score_sleep(_nullable(today["sleep_duration_hrs"]))
    hr_s = score_hr(
        _nullable(today["resting_heart_rate"]),
        past_28["resting_heart_rate"].dropna(),
    )
    load_s = score_load(float(past_7["workout_minutes"].fillna(0).sum()))
    consistency_s = score_consistency(past_14["sleep_start"].dropna())

    components = {"sleep": sleep_s, "hr": hr_s, "load": load_s, "consistency": consistency_s}
    weights = {
        "sleep": settings.readiness_weight_sleep,
        "hr": settings.readiness_weight_hr,
        "load": settings.readiness_weight_load,
        "consistency": settings.readiness_weight_consistency,
    }

    final_score, explanation = _weighted_score(components, weights)

    return {
        "date": d,
        "score": final_score,
        "sleep_score": sleep_s,
        "hr_score": hr_s,
        "load_score": load_s,
        "consistency_score": consistency_s,
        "explanation": explanation,
    }


def _weighted_score(
    components: dict[str, Optional[float]],
    weights: dict[str, float],
) -> tuple[float, str]:
    available = {k: v for k, v in components.items() if v is not None}

    if not available:
        return 50.0, "Insufficient data to compute readiness score."

    total_weight = sum(weights[k] for k in available)
    score = sum(v * weights[k] / total_weight for k, v in available.items())
    return round(score, 1), _build_explanation(components, score)


def _build_explanation(components: dict[str, Optional[float]], score: float) -> str:
    thresholds = {
        "sleep": ("sleep", 75, 50),
        "hr": ("resting HR", 75, 50),
        "load": ("training load", 82, 58),
        "consistency": ("sleep schedule", 75, 50),
    }
    good, poor = [], []
    for key, (label, good_t, poor_t) in thresholds.items():
        v = components.get(key)
        if v is None:
            continue
        if v >= good_t:
            good.append(label)
        elif v < poor_t:
            poor.append(label)

    prefix = "High readiness" if score >= 80 else ("Moderate readiness" if score >= 60 else "Low readiness")
    parts = []
    if good:
        parts.append(f"strong {', '.join(good)}")
    if poor:
        parts.append(f"limited by {', '.join(poor)}")

    return f"{prefix}: {'; '.join(parts)}." if parts else f"{prefix}: score {round(score)}/100."


def _upsert(db: Session, rows: list[dict]) -> None:
    stmt = insert(ReadinessScore).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date"],
        set_={
            "score": stmt.excluded.score,
            "sleep_score": stmt.excluded.sleep_score,
            "hr_score": stmt.excluded.hr_score,
            "load_score": stmt.excluded.load_score,
            "consistency_score": stmt.excluded.consistency_score,
            "explanation": stmt.excluded.explanation,
        },
    )
    db.execute(stmt)
    db.commit()


def _nullable(val) -> Optional[float]:
    """Convert pandas NA/NaN to Python None."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return float(val) if val is not None else None
