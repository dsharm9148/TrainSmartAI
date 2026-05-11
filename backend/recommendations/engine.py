"""
Rule-based recommendation engine.

Each rule is a plain function that inspects today's daily_features snapshot
and recent history, then returns a Recommendation dict or None. Rules fire
independently — multiple can trigger on the same day.

Categories and priorities:
  recovery  — when physiological signals suggest the body needs rest
  workout   — when the user is ready to train or needs a push
  sleep     — actionable sleep hygiene suggestions
  habit     — low-priority lifestyle nudges

Priority: 1 = act on this today, 2 = worth doing, 3 = nice to have.

Public API:
  generate_recommendations(db, for_date, lookback_days) -> int
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from backend.db.models import DailyFeatures, ReadinessScore, Recommendation

LOOKBACK_DAYS = 14


@dataclass
class _Rec:
    category: str
    priority: int
    text: str
    reasoning: str


# ─── Public API ───────────────────────────────────────────────────────────────


def generate_recommendations(
    db: Session,
    for_date: Optional[date] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> int:
    """
    Generate rule-based recommendations for for_date.

    Deletes existing recommendations for that date before writing new ones.
    Returns the number of recommendations written.
    """
    if for_date is None:
        from datetime import date as _d
        for_date = _d.today()

    start = for_date - timedelta(days=lookback_days - 1)
    df = _load_features(db, start, for_date)
    readiness = _load_readiness(db, for_date)

    today = _today_row(df, for_date)

    rules = [
        _rule_high_rhr_recovery,
        _rule_low_readiness_rest,
        _rule_high_readiness_train,
        _rule_low_sleep_tonight,
        _rule_sleep_deficit,
        _rule_no_workout_streak,
        _rule_overtraining_warning,
        _rule_consistent_bedtime,
        _rule_step_goal_nudge,
        _rule_post_anomaly_recovery,
    ]

    recs: list[_Rec] = []
    for rule in rules:
        try:
            rec = rule(today, df, readiness)
            if rec is not None:
                recs.append(rec)
        except Exception:
            pass

    if not recs:
        return 0

    db.query(Recommendation).filter(Recommendation.date == for_date).delete()
    for r in recs:
        db.add(Recommendation(
            date=for_date,
            category=r.category,
            priority=r.priority,
            text=r.text,
            reasoning=r.reasoning,
        ))
    db.commit()
    return len(recs)


# ─── Rules ────────────────────────────────────────────────────────────────────


def _rule_high_rhr_recovery(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when today's RHR is ≥ 5 bpm above the 14-day rolling mean."""
    if today is None:
        return None
    rhr = _val(today, "resting_heart_rate")
    if rhr is None:
        return None

    history = df[df["date"] < today["date"]]["resting_heart_rate"].dropna()
    if len(history) < 3:
        return None

    baseline = float(history.mean())
    elevation = rhr - baseline
    if elevation < 5.0:
        return None

    return _Rec(
        category="recovery",
        priority=1,
        text=f"Your resting HR is {elevation:.0f} bpm above your baseline today. Prioritise rest, hydration, and light movement over intense training.",
        reasoning=f"rhr={rhr:.0f}, baseline={baseline:.0f}, elevation={elevation:.1f} >= 5.0",
    )


def _rule_low_readiness_rest(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when today's readiness score is below 55."""
    if readiness is None or readiness >= 55:
        return None
    return _Rec(
        category="recovery",
        priority=1,
        text=f"Your readiness score is {readiness:.0f}/100 — your body is asking for a lighter day. Skip high-intensity work and focus on recovery: walk, stretch, or rest.",
        reasoning=f"readiness={readiness:.1f} < 55",
    )


def _rule_high_readiness_train(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when readiness is ≥ 80 and there was no workout in the last 2 days."""
    if readiness is None or readiness < 80:
        return None

    recent = df[df["date"] <= today["date"]].tail(3) if today is not None else df.tail(3)
    recent_workouts = recent["workout_count"].fillna(0).sum()
    if recent_workouts > 0:
        return None

    return _Rec(
        category="workout",
        priority=2,
        text=f"Readiness is high ({readiness:.0f}/100) and you haven't trained in the last 2 days — a great opportunity for a quality workout today.",
        reasoning=f"readiness={readiness:.1f} >= 80, no workout in last 3 rows",
    )


def _rule_low_sleep_tonight(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when today's sleep was below 6 hours."""
    if today is None:
        return None
    sleep = _val(today, "sleep_duration_hrs")
    if sleep is None or sleep >= 6.0:
        return None

    return _Rec(
        category="sleep",
        priority=1,
        text=f"You slept {sleep:.1f} hrs last night. Aim to be in bed 30–60 minutes earlier tonight to reduce sleep debt.",
        reasoning=f"sleep_duration_hrs={sleep:.2f} < 6.0",
    )


def _rule_sleep_deficit(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when the 7-day average sleep is below 6.5 hours."""
    recent = df.tail(7)
    sleep = recent["sleep_duration_hrs"].dropna()
    if len(sleep) < 4:
        return None

    avg = float(sleep.mean())
    if avg >= 6.5:
        return None

    deficit = 7.0 - avg
    return _Rec(
        category="sleep",
        priority=2,
        text=f"Your 7-day sleep average is {avg:.1f} hrs — {deficit:.1f} hrs short of the 7-hr target. Consistent early bedtimes will have the biggest impact on your readiness scores.",
        reasoning=f"7-day avg sleep={avg:.2f} < 6.5",
    )


def _rule_no_workout_streak(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when there have been no workouts in the last 5 days."""
    recent = df.tail(5)
    if recent.empty:
        return None
    if recent["workout_count"].notna().sum() > 0:
        return None

    return _Rec(
        category="workout",
        priority=2,
        text="You haven't logged a workout in 5 days. Even a 20-minute walk counts — consistency beats intensity over time.",
        reasoning="workout_count is null for all of last 5 days",
    )


def _rule_overtraining_warning(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when 7-day workout volume exceeds 250 minutes AND readiness is below 65."""
    recent = df.tail(7)
    total_min = float(recent["workout_minutes"].fillna(0).sum())
    if total_min < 250:
        return None
    if readiness is not None and readiness >= 65:
        return None

    return _Rec(
        category="recovery",
        priority=1,
        text=f"You've logged {total_min:.0f} workout minutes in the last 7 days — that's a heavy load. Consider a rest or deload day to avoid accumulated fatigue.",
        reasoning=f"7-day workout_minutes={total_min:.0f} >= 250, readiness={readiness}",
    )


def _rule_consistent_bedtime(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when sleep-start std dev > 1.5 h over 14 days — suggest consistent bedtime."""
    starts = df["sleep_start"].dropna()
    if len(starts) < 7:
        return None

    hours = pd.to_datetime(starts).apply(lambda dt: dt.hour + dt.minute / 60.0)
    std = float(hours.std())
    if std < 1.5:
        return None

    return _Rec(
        category="habit",
        priority=3,
        text=f"Your bedtime varies by over {std:.1f} hours night to night. A consistent sleep schedule — even on weekends — is one of the highest-leverage habits for recovery.",
        reasoning=f"sleep_start std={std:.2f} h >= 1.5",
    )


def _rule_step_goal_nudge(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """Fire when 7-day average steps is below 6,000."""
    recent = df.tail(7)
    steps = recent["steps"].dropna()
    if len(steps) < 4:
        return None

    avg = float(steps.mean())
    if avg >= 6000:
        return None

    return _Rec(
        category="habit",
        priority=3,
        text=f"Your 7-day step average is {avg:,.0f}/day. Adding a 15-minute walk after meals is one of the easiest ways to reach 8,000 steps without structured exercise.",
        reasoning=f"7-day avg steps={avg:.0f} < 6000",
    )


def _rule_post_anomaly_recovery(
    today: Optional[pd.Series],
    df: pd.DataFrame,
    readiness: Optional[float],
) -> Optional[_Rec]:
    """
    Fire when yesterday had very low steps (< 3,000) combined with elevated RHR —
    a signature of illness, travel, or high stress.
    """
    yesterday = df[df["date"] == today["date"] - timedelta(days=1)] if today is not None else pd.DataFrame()
    if yesterday.empty:
        return None

    y = yesterday.iloc[0]
    steps = _val(y, "steps")
    rhr = _val(y, "resting_heart_rate")

    if steps is None or rhr is None:
        return None
    if steps >= 3000:
        return None

    # Check if RHR was elevated vs baseline
    history = df[df["date"] < y["date"]]["resting_heart_rate"].dropna()
    if len(history) < 3:
        return None
    if rhr - float(history.mean()) < 3.0:
        return None

    return _Rec(
        category="recovery",
        priority=2,
        text="Yesterday's data suggests illness, travel, or high stress (low steps + elevated HR). Ease back into training gradually over the next 1–2 days.",
        reasoning=f"yesterday steps={steps}, rhr={rhr:.0f}, elevated vs baseline",
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_features(db: Session, start: date, end: date) -> pd.DataFrame:
    rows = (
        db.query(DailyFeatures)
        .filter(DailyFeatures.date >= start, DailyFeatures.date <= end)
        .order_by(DailyFeatures.date)
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=[
            "date", "steps", "resting_heart_rate", "sleep_duration_hrs",
            "sleep_start", "workout_count", "workout_minutes",
        ])
    return pd.DataFrame([
        {
            "date": r.date,
            "steps": r.steps,
            "resting_heart_rate": r.resting_heart_rate,
            "sleep_duration_hrs": r.sleep_duration_hrs,
            "sleep_start": r.sleep_start,
            "workout_count": r.workout_count,
            "workout_minutes": r.workout_minutes,
        }
        for r in rows
    ])


def _load_readiness(db: Session, for_date: date) -> Optional[float]:
    row = db.query(ReadinessScore).filter(ReadinessScore.date == for_date).first()
    return float(row.score) if row else None


def _today_row(df: pd.DataFrame, for_date: date) -> Optional[pd.Series]:
    mask = df["date"] == for_date
    if not mask.any():
        return None
    return df[mask].iloc[0]


def _val(row: pd.Series, col: str) -> Optional[float]:
    """Return float value or None for NaN/None."""
    try:
        v = row[col]
        if pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError, KeyError):
        return None
