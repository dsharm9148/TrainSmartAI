"""
Converts DB rows into LangChain Document objects for RAG indexing.

Each document type carries typed metadata so downstream filters can
narrow retrieval to a specific date range or data category.
"""
from __future__ import annotations

from langchain_core.documents import Document
from sqlalchemy.orm import Session

from backend.db.models import (
    DailyFeatures,
    Insight,
    ReadinessScore,
    Recommendation,
    WeeklySummary,
)


def build_daily_doc(row: DailyFeatures) -> Document:
    parts = [f"Daily summary for {row.date}:"]
    if row.steps is not None:
        parts.append(f"steps={row.steps:,}")
    if row.sleep_duration_hrs is not None:
        parts.append(f"sleep={row.sleep_duration_hrs:.1f}h")
    if row.resting_heart_rate is not None:
        parts.append(f"resting HR={row.resting_heart_rate:.0f} bpm")
    if row.avg_heart_rate is not None:
        parts.append(f"avg HR={row.avg_heart_rate:.0f} bpm")
    if row.workout_count:
        wparts = [f"{row.workout_count} session(s)"]
        if row.workout_minutes:
            wparts.append(f"{row.workout_minutes:.0f} min")
        if row.workout_calories:
            wparts.append(f"{row.workout_calories:.0f} cal")
        parts.append("workout: " + ", ".join(wparts))
    if row.active_energy_kcal is not None:
        parts.append(f"active energy={row.active_energy_kcal:.0f} kcal")
    return Document(
        page_content=" | ".join(parts),
        metadata={"type": "daily", "date": str(row.date)},
    )


def build_weekly_doc(row: WeeklySummary) -> Document:
    parts = [f"Weekly summary {row.week_start} to {row.week_end}:"]
    if row.avg_daily_steps is not None:
        parts.append(f"avg {row.avg_daily_steps:,.0f} steps/day")
    if row.avg_sleep_hrs is not None:
        parts.append(f"avg sleep {row.avg_sleep_hrs:.1f}h")
    if row.sleep_consistency_score is not None:
        parts.append(f"sleep consistency {row.sleep_consistency_score:.0f}/100")
    if row.avg_resting_hr is not None:
        parts.append(f"avg resting HR {row.avg_resting_hr:.0f} bpm")
    if row.workout_days is not None:
        parts.append(f"{row.workout_days} workout day(s)")
    if row.total_workout_minutes is not None:
        parts.append(f"{row.total_workout_minutes:.0f} workout min total")
    if row.avg_readiness_score is not None:
        parts.append(f"avg readiness {row.avg_readiness_score:.0f}/100")
    return Document(
        page_content=" | ".join(parts),
        metadata={"type": "weekly", "week_start": str(row.week_start)},
    )


def build_readiness_doc(row: ReadinessScore) -> Document:
    parts = [f"Readiness on {row.date}: {row.score:.0f}/100"]
    components = []
    if row.sleep_score is not None:
        components.append(f"sleep={row.sleep_score:.0f}")
    if row.hr_score is not None:
        components.append(f"HR={row.hr_score:.0f}")
    if row.load_score is not None:
        components.append(f"load={row.load_score:.0f}")
    if row.consistency_score is not None:
        components.append(f"consistency={row.consistency_score:.0f}")
    if components:
        parts.append("components: " + ", ".join(components))
    if row.explanation:
        parts.append(row.explanation)
    return Document(
        page_content=". ".join(parts),
        metadata={"type": "readiness", "date": str(row.date)},
    )


def build_insight_doc(row: Insight) -> Document:
    prefix = f"Health insight ({row.generated_for})"
    if row.insight_type:
        prefix += f" [{row.insight_type}]"
    return Document(
        page_content=f"{prefix}: {row.text}",
        metadata={
            "type": "insight",
            "date": str(row.generated_for),
            "insight_type": row.insight_type or "",
        },
    )


def build_recommendation_doc(row: Recommendation) -> Document:
    header = f"Recommendation for {row.date}"
    if row.category:
        header += f" [{row.category}]"
    if row.priority:
        header += f" priority {row.priority}"
    text = f"{header}: {row.text}"
    if row.reasoning:
        text += f" (Reason: {row.reasoning})"
    return Document(
        page_content=text,
        metadata={
            "type": "recommendation",
            "date": str(row.date),
            "category": row.category or "",
        },
    )


def load_all_documents(db: Session) -> list[Document]:
    """Load every health record from the DB and return as LangChain Documents."""
    docs: list[Document] = []
    for row in db.query(DailyFeatures).order_by(DailyFeatures.date).all():
        docs.append(build_daily_doc(row))
    for row in db.query(WeeklySummary).order_by(WeeklySummary.week_start).all():
        docs.append(build_weekly_doc(row))
    for row in db.query(ReadinessScore).order_by(ReadinessScore.date).all():
        docs.append(build_readiness_doc(row))
    for row in db.query(Insight).order_by(Insight.generated_for).all():
        docs.append(build_insight_doc(row))
    for row in db.query(Recommendation).order_by(Recommendation.date).all():
        docs.append(build_recommendation_doc(row))
    return docs
