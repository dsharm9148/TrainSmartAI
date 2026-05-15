from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.schemas import ReadinessOut
from backend.db.models import ReadinessScore
from backend.db.session import get_db
from backend.scoring.readiness import compute_readiness_scores

router = APIRouter()


@router.post("/readiness/compute", tags=["scoring"])
def trigger_readiness(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD). Defaults to all available data."),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
) -> dict:
    """Compute (or recompute) readiness scores from daily_features data."""
    count = compute_readiness_scores(db, start_date, end_date)
    return {"days_scored": count, "message": f"Readiness computed for {count} days."}


@router.get("/readiness", response_model=list[ReadinessOut], tags=["scoring"])
def get_readiness(
    start_date: Optional[date] = Query(None, description="Start date inclusive (YYYY-MM-DD). Defaults to 30 days ago."),
    end_date: Optional[date] = Query(None, description="End date inclusive (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
) -> list[ReadinessOut]:
    """Return readiness scores for the requested date range, sorted oldest-first."""
    from datetime import date as date_cls
    from datetime import timedelta
    if end_date is None:
        end_date = date_cls.today()
    if start_date is None:
        start_date = end_date - timedelta(days=29)

    return (
        db.query(ReadinessScore)
        .filter(ReadinessScore.date >= start_date, ReadinessScore.date <= end_date)
        .order_by(ReadinessScore.date)
        .all()
    )
