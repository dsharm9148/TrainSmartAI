"""
Daily features computation.

Reads raw health_records and aggregates them into one row per calendar day
in the daily_features table.

Attribution rules:
  Steps       — attributed to the day their start_date falls on
  Heart rate  — averaged across all readings on start_date's day
  Resting HR  — minimum value on start_date's day (Apple emits one per day,
                but take min to be safe if multiple exist)
  Sleep       — attributed to the day the record's end_date falls on
                (i.e. the wake-up day). This matches how Whoop assigns
                recovery: last night's sleep belongs to today's readiness.
  Workouts    — attributed to the day their start_date falls on

Missing metrics for a given day are stored as None, not 0.
None vs 0 is a meaningful distinction: "no data" is different from "zero steps".
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.db.models import DailyFeatures, HealthRecord
from backend.ingestion.parser import HEART_RATE, RESTING_HR, SLEEP, STEP_COUNT

WORKOUT_PREFIX = "HKWorkoutActivityType"


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_daily_features(
    db: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> int:
    """
    Compute and upsert daily feature rows for all days with health data.

    If start_date / end_date are given, only those days are (re)computed.
    Returns the number of days processed.
    """
    steps_df = _load_type(db, STEP_COUNT)
    hr_df = _load_type(db, HEART_RATE)
    rhr_df = _load_type(db, RESTING_HR)
    sleep_df = _load_type(db, SLEEP)
    workout_df = _load_workouts(db)

    dates = _determine_dates(
        [steps_df, hr_df, rhr_df, sleep_df, workout_df],
        start_date,
        end_date,
    )

    if not dates:
        return 0

    rows = [
        _compute_day(d, steps_df, hr_df, rhr_df, sleep_df, workout_df)
        for d in dates
    ]
    _upsert_rows(db, rows)
    return len(rows)


# ─── Data loading ─────────────────────────────────────────────────────────────


def _load_type(db: Session, record_type: str) -> pd.DataFrame:
    """Load all records of a given type into a DataFrame."""
    records = (
        db.query(HealthRecord)
        .filter(HealthRecord.record_type == record_type)
        .all()
    )
    if not records:
        return pd.DataFrame(columns=["value", "start_date", "end_date"])

    return pd.DataFrame(
        [{"value": r.value, "start_date": r.start_date, "end_date": r.end_date}
         for r in records]
    )


def _load_workouts(db: Session) -> pd.DataFrame:
    """Load all workout records (all HKWorkoutActivityType* types)."""
    records = (
        db.query(HealthRecord)
        .filter(HealthRecord.record_type.like(f"{WORKOUT_PREFIX}%"))
        .all()
    )
    if not records:
        return pd.DataFrame(columns=["value", "start_date", "end_date"])

    return pd.DataFrame(
        [{"value": r.value, "start_date": r.start_date, "end_date": r.end_date}
         for r in records]
    )


# ─── Date range determination ─────────────────────────────────────────────────


def _determine_dates(
    dfs: list[pd.DataFrame],
    start_date: Optional[date],
    end_date: Optional[date],
) -> list[date]:
    """
    Return the sorted list of calendar dates to process.

    Derived from the union of:
      - start_date of all activity records (steps, HR, workouts)
      - end_date of sleep records (wake-up day attribution)
    """
    all_dates: set[date] = set()

    for df in dfs:
        if df.empty:
            continue
        if "start_date" in df.columns and not df["start_date"].isna().all():
            all_dates.update(df["start_date"].dt.date.dropna().unique())

    # Sleep is attributed to its end_date (wake-up day)
    sleep_df = dfs[3] if len(dfs) > 3 else pd.DataFrame()
    if not sleep_df.empty and "end_date" in sleep_df.columns:
        all_dates.update(sleep_df["end_date"].dt.date.dropna().unique())

    if not all_dates:
        return []

    dates = sorted(all_dates)

    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    return dates


# ─── Per-day aggregation ──────────────────────────────────────────────────────


def _compute_day(
    d: date,
    steps_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    rhr_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    workout_df: pd.DataFrame,
) -> dict:
    """Compute all feature columns for a single calendar day."""

    # Steps: sum all records starting on this day
    day_steps = _by_start_date(steps_df, d)
    steps = int(day_steps["value"].sum()) if not day_steps.empty else None

    # Average heart rate across all readings on this day
    day_hr = _by_start_date(hr_df, d)
    avg_hr = round(float(day_hr["value"].mean()), 1) if not day_hr.empty else None

    # Resting HR: Apple emits one per day; take min if there are duplicates
    day_rhr = _by_start_date(rhr_df, d)
    resting_hr = round(float(day_rhr["value"].min()), 1) if not day_rhr.empty else None

    # Sleep: records whose end_date falls on this day (wake-up day attribution)
    day_sleep = _sleep_for_day(sleep_df, d)
    sleep_duration = round(float(day_sleep["value"].sum()), 2) if not day_sleep.empty else None
    sleep_start = day_sleep["start_date"].min() if not day_sleep.empty else None
    sleep_end = day_sleep["end_date"].max() if not day_sleep.empty else None

    # Workouts: records starting on this day
    day_wo = _by_start_date(workout_df, d)
    workout_count = int(len(day_wo)) if not day_wo.empty else None
    workout_minutes = round(float(day_wo["value"].sum()), 1) if not day_wo.empty else None

    return {
        "date": d,
        "steps": steps,
        "avg_heart_rate": avg_hr,
        "resting_heart_rate": resting_hr,
        "sleep_duration_hrs": sleep_duration,
        "sleep_start": _none_if_nat(sleep_start),
        "sleep_end": _none_if_nat(sleep_end),
        "workout_count": workout_count,
        "workout_minutes": workout_minutes,
        "workout_calories": None,   # not tracked in current ingestion
        "active_energy_kcal": None, # not tracked in current ingestion
    }


# ─── DataFrame filters ────────────────────────────────────────────────────────


def _by_start_date(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Rows where start_date falls on the given calendar date."""
    if df.empty:
        return df
    return df[df["start_date"].dt.date == d]


def _sleep_for_day(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """
    Sleep records attributed to day d.

    A sleep record belongs to the day it ENDS on — the wake-up day.
    This mirrors how Whoop assigns recovery: yesterday's sleep
    contributes to today's readiness score.
    """
    if df.empty:
        return df
    return df[df["end_date"].dt.date == d]


# ─── Database write ───────────────────────────────────────────────────────────


def _upsert_rows(db: Session, rows: list[dict]) -> None:
    """Upsert daily feature rows — insert new, update existing."""
    if not rows:
        return

    stmt = insert(DailyFeatures).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date"],
        set_={
            "steps": stmt.excluded.steps,
            "avg_heart_rate": stmt.excluded.avg_heart_rate,
            "resting_heart_rate": stmt.excluded.resting_heart_rate,
            "sleep_duration_hrs": stmt.excluded.sleep_duration_hrs,
            "sleep_start": stmt.excluded.sleep_start,
            "sleep_end": stmt.excluded.sleep_end,
            "workout_count": stmt.excluded.workout_count,
            "workout_minutes": stmt.excluded.workout_minutes,
            "workout_calories": stmt.excluded.workout_calories,
            "active_energy_kcal": stmt.excluded.active_energy_kcal,
        },
    )
    db.execute(stmt)
    db.commit()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _none_if_nat(value) -> Optional[object]:
    """Convert pandas NaT to None so SQLAlchemy doesn't choke."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value
