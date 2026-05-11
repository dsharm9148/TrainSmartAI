"""
Insights engine.

Analyses the last N days of daily_features and writes plain-English
observations to the insights table. Each call to generate_insights()
replaces any existing insights for that date, so re-running is safe.

Seven insight types:
  sleep_activity         — sleep duration vs same-day step count (Pearson r)
  hr_trend               — linear slope of resting HR over the window
  sleep_quality          — average sleep + early vs recent half comparison
  workout_pattern        — frequency and total volume
  post_workout_recovery  — RHR elevation the day after a workout
  weekend_sleep          — weekend vs weekday sleep difference
  steps_trend            — step count trend and average vs 8k target
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.db.models import DailyFeatures, Insight

LOOKBACK_DAYS = 30
MIN_ROWS = 5  # Need at least this many days before generating anything


# ─── Public API ───────────────────────────────────────────────────────────────


def generate_insights(
    db: Session,
    for_date: Optional[date] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> int:
    """
    Generate insights from the lookback_days ending on for_date (inclusive).

    Deletes any existing insights for for_date before writing new ones.
    Returns the number of insights written.
    """
    if for_date is None:
        from datetime import date as _d
        for_date = _d.today()

    start = for_date - timedelta(days=lookback_days - 1)
    df = _load_features(db, start, for_date)

    if len(df) < MIN_ROWS:
        return 0

    generators = [
        ("sleep_activity", _insight_sleep_activity),
        ("hr_trend", _insight_rhr_trend),
        ("sleep_quality", _insight_sleep_trend),
        ("workout_pattern", _insight_workout_pattern),
        ("post_workout_recovery", _insight_post_workout_rhr),
        ("weekend_sleep", _insight_weekend_sleep),
        ("steps_trend", _insight_steps_trend),
    ]

    rows = []
    for insight_type, fn in generators:
        try:
            text = fn(df)
            if text:
                rows.append({"insight_type": insight_type, "text": text})
        except Exception:
            pass  # Never let one broken generator halt the whole run

    if not rows:
        return 0

    db.query(Insight).filter(Insight.generated_for == for_date).delete()
    for row in rows:
        db.add(Insight(generated_for=for_date, **row))
    db.commit()

    return len(rows)


# ─── Individual insight generators ────────────────────────────────────────────


def _insight_sleep_activity(df: pd.DataFrame) -> Optional[str]:
    """Pearson correlation between sleep hours and same-day step count."""
    d = df[["sleep_duration_hrs", "steps"]].dropna()
    if len(d) < 10:
        return None

    r = float(d["sleep_duration_hrs"].corr(d["steps"]))
    if not np.isfinite(r) or abs(r) < 0.25:
        return None

    direction = "higher" if r > 0 else "lower"
    return (
        f"On days following better sleep, your step count tends to be {direction} "
        f"(r={r:.2f} across {len(d)} days). "
        f"{'Protecting sleep may directly boost your daily activity.' if r > 0 else 'Other factors may be driving your activity levels.'}"
    )


def _insight_rhr_trend(df: pd.DataFrame) -> Optional[str]:
    """Linear trend in resting heart rate over the window."""
    rhr = df["resting_heart_rate"].dropna()
    if len(rhr) < 7:
        return None

    x = np.arange(len(rhr), dtype=float)
    slope, _ = np.polyfit(x, rhr.values.astype(float), 1)
    total_change = slope * (len(rhr) - 1)

    if abs(total_change) < 1.5:
        avg = float(rhr.mean())
        return f"Your resting HR has been stable at {avg:.0f} bpm over the last {len(rhr)} days."

    direction = "dropped" if total_change < 0 else "risen"
    note = "a sign of improving cardiovascular fitness" if total_change < 0 else "consider adding more recovery days"
    return f"Your resting HR has {direction} {abs(total_change):.1f} bpm over the last {len(rhr)} days — {note}."


def _insight_sleep_trend(df: pd.DataFrame) -> Optional[str]:
    """Average sleep duration with early-vs-recent comparison if ≥ 14 days."""
    sleep = df["sleep_duration_hrs"].dropna()
    if len(sleep) < 7:
        return None

    avg = float(sleep.mean())

    if len(sleep) >= 14:
        mid = len(sleep) // 2
        early = float(sleep.iloc[:mid].mean())
        recent = float(sleep.iloc[mid:].mean())
        change = recent - early
        if abs(change) >= 0.4:
            direction = "improving" if change > 0 else "declining"
            return (
                f"Your sleep is {direction}: {recent:.1f} hrs recently vs "
                f"{early:.1f} hrs earlier in this {len(sleep)}-day window "
                f"(overall avg {avg:.1f} hrs)."
            )

    qualifier = "solid" if avg >= 7.5 else ("adequate" if avg >= 6.5 else "below the recommended 7–9 hrs")
    return f"You're averaging {avg:.1f} hrs of sleep per night over the last {len(sleep)} days — {qualifier}."


def _insight_workout_pattern(df: pd.DataFrame) -> Optional[str]:
    """Workout frequency and total volume over the window."""
    n_days = len(df)
    workout_days = int(df["workout_count"].notna().sum())

    if workout_days == 0:
        return f"No workouts recorded in the last {n_days} days."

    total_min = float(df["workout_minutes"].fillna(0).sum())
    freq = workout_days / n_days * 7
    return (
        f"You've worked out {workout_days} times in the last {n_days} days "
        f"({freq:.1f}x/week), logging {total_min:.0f} total minutes."
    )


def _insight_post_workout_rhr(df: pd.DataFrame) -> Optional[str]:
    """Compares RHR on the day after a workout vs day after rest."""
    d = df[["workout_count", "resting_heart_rate"]].copy()
    d["prev_workout"] = d["workout_count"].shift(1).fillna(0) > 0
    d = d.dropna(subset=["resting_heart_rate"])

    after_workout = d[d["prev_workout"]]["resting_heart_rate"]
    after_rest = d[~d["prev_workout"]]["resting_heart_rate"]

    if len(after_workout) < 3 or len(after_rest) < 3:
        return None

    diff = float(after_workout.mean()) - float(after_rest.mean())
    if abs(diff) < 1.5:
        return None

    if diff > 0:
        return (
            f"Your resting HR is {diff:.1f} bpm higher the day after a workout "
            f"({after_workout.mean():.0f} vs {after_rest.mean():.0f} bpm on rest days) — normal acute recovery stress."
        )
    return (
        f"Your resting HR is {abs(diff):.1f} bpm lower the day after a workout — "
        f"you recover quickly and adapt well to training."
    )


def _insight_weekend_sleep(df: pd.DataFrame) -> Optional[str]:
    """Weekend vs weekday sleep duration difference."""
    d = df[df["sleep_duration_hrs"].notna()].copy()
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek  # 0=Mon, 6=Sun

    weekend = d[d["dow"] >= 5]["sleep_duration_hrs"]
    weekday = d[d["dow"] < 5]["sleep_duration_hrs"]

    if len(weekend) < 2 or len(weekday) < 5:
        return None

    diff = float(weekend.mean()) - float(weekday.mean())
    if abs(diff) < 0.4:
        return None

    direction = "longer" if diff > 0 else "shorter"
    note = "Social jet lag can disrupt your circadian rhythm." if diff > 1.0 else ""
    return (
        f"You sleep {abs(diff):.1f} hrs {direction} on weekends "
        f"({weekend.mean():.1f} hrs) vs weekdays ({weekday.mean():.1f} hrs). {note}"
    ).strip()


def _insight_steps_trend(df: pd.DataFrame) -> Optional[str]:
    """Linear step count trend and average vs 8k/day target."""
    steps = df["steps"].dropna()
    if len(steps) < 7:
        return None

    avg = float(steps.mean())
    x = np.arange(len(steps), dtype=float)
    slope, _ = np.polyfit(x, steps.values.astype(float), 1)
    total_change = slope * (len(steps) - 1)

    if abs(total_change) > 1000:
        direction = "trending up" if total_change > 0 else "trending down"
        return (
            f"Your daily steps are {direction} "
            f"(+{total_change:+.0f} over the period, avg {avg:,.0f}/day)."
        )

    qualifier = "above the 8,000-step target" if avg >= 8000 else "below the 8,000-step daily target"
    return f"Averaging {avg:,.0f} steps/day over the last {len(steps)} days — {qualifier}."


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_features(db: Session, start: date, end: date) -> pd.DataFrame:
    rows = (
        db.query(DailyFeatures)
        .filter(DailyFeatures.date >= start, DailyFeatures.date <= end)
        .order_by(DailyFeatures.date)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "date": r.date,
            "steps": r.steps,
            "resting_heart_rate": r.resting_heart_rate,
            "sleep_duration_hrs": r.sleep_duration_hrs,
            "workout_count": r.workout_count,
            "workout_minutes": r.workout_minutes,
        }
        for r in rows
    ])
