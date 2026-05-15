"""
Weekly summary computation.

Aggregates daily_features and readiness_scores into one row per ISO week
(Monday–Sunday). Safe to re-run: existing rows are updated in place.

Public API:
  compute_weekly_summaries(db) -> int
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.db.models import DailyFeatures, ReadinessScore, WeeklySummary


def compute_weekly_summaries(db: Session) -> int:
    """
    Build or refresh weekly_summaries from all available daily data.

    Groups days by ISO week (week_start = Monday). Upserts one row per week.
    Returns the number of weeks written.
    """
    feature_rows = db.query(DailyFeatures).order_by(DailyFeatures.date).all()
    if not feature_rows:
        return 0

    df = pd.DataFrame([
        {
            "date": r.date,
            "steps": r.steps,
            "resting_heart_rate": r.resting_heart_rate,
            "sleep_duration_hrs": r.sleep_duration_hrs,
            "workout_count": r.workout_count,
            "workout_minutes": r.workout_minutes,
        }
        for r in feature_rows
    ])

    readiness_rows = db.query(ReadinessScore).all()
    rdf = (
        pd.DataFrame([{"date": r.date, "score": r.score} for r in readiness_rows])
        if readiness_rows
        else pd.DataFrame(columns=["date", "score"])
    )

    df["week_start"] = df["date"].apply(lambda d: d - timedelta(days=d.weekday()))

    count = 0
    for week_start, group in df.groupby("week_start"):
        week_end = week_start + timedelta(days=6)

        avg_steps = _nanmean(group["steps"])
        avg_sleep = _nanmean(group["sleep_duration_hrs"])
        sleep_consistency = _sleep_consistency(group["sleep_duration_hrs"])
        avg_rhr = _nanmean(group["resting_heart_rate"])
        total_workout_min = float(group["workout_minutes"].fillna(0).sum())
        workout_days = int((group["workout_count"].fillna(0) > 0).sum())

        if not rdf.empty:
            mask = (rdf["date"] >= week_start) & (rdf["date"] <= week_end)
            week_scores = rdf[mask]["score"]
            avg_readiness: Optional[float] = (
                float(week_scores.mean()) if len(week_scores) > 0 else None
            )
        else:
            avg_readiness = None

        row = db.query(WeeklySummary).filter(WeeklySummary.week_start == week_start).first()
        if row is None:
            row = WeeklySummary(week_start=week_start)
            db.add(row)

        row.week_end = week_end
        row.avg_daily_steps = avg_steps
        row.avg_sleep_hrs = avg_sleep
        row.sleep_consistency_score = sleep_consistency
        row.avg_resting_hr = avg_rhr
        row.total_workout_minutes = total_workout_min if total_workout_min > 0 else None
        row.workout_days = workout_days
        row.avg_readiness_score = avg_readiness

        count += 1

    db.commit()
    return count


def _nanmean(series: pd.Series) -> Optional[float]:
    """Return float mean ignoring NaN, or None if all values are NaN."""
    valid = series.dropna()
    return float(valid.mean()) if len(valid) > 0 else None


def _sleep_consistency(series: pd.Series) -> Optional[float]:
    """
    Convert sleep-duration std dev into a 0-100 consistency score.

    0 h std → 100 (perfectly consistent). 4 h std → 0 (wildly irregular).
    Returns None when fewer than 2 non-null readings exist.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return None
    std = float(valid.std())
    return max(0.0, 100.0 - std * 25.0)
