from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.analytics.insights import LOOKBACK_DAYS, generate_insights
from backend.api.schemas import InsightOut
from backend.db.models import Insight
from backend.db.session import get_db

router = APIRouter()


@router.post("/insights/generate", tags=["insights"])
def trigger_insights(
    for_date: Optional[date] = Query(None, description="As-of date (YYYY-MM-DD). Defaults to today."),
    lookback_days: int = Query(LOOKBACK_DAYS, ge=7, le=90, description="Days of history to analyse."),
    db: Session = Depends(get_db),
) -> dict:
    """Generate plain-English insights from the last lookback_days of daily feature data."""
    count = generate_insights(db, for_date, lookback_days)
    return {"insights_generated": count, "message": f"Generated {count} insights."}


@router.get("/insights", response_model=list[InsightOut], tags=["insights"])
def get_insights(
    for_date: Optional[date] = Query(None, description="Filter by generated_for date (YYYY-MM-DD)."),
    db: Session = Depends(get_db),
) -> list[InsightOut]:
    """Return stored insights, optionally filtered to a specific generated_for date."""
    q = db.query(Insight)
    if for_date:
        q = q.filter(Insight.generated_for == for_date)
    return q.order_by(Insight.generated_for.desc(), Insight.id).all()
