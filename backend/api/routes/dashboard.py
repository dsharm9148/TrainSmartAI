from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.schemas import DailyFeaturesOut, WeeklySummaryOut
from backend.db.models import DailyFeatures, WeeklySummary
from backend.db.session import get_db

router = APIRouter()


@router.get("/daily", response_model=list[DailyFeaturesOut], tags=["dashboard"])
def get_daily(
    start_date: Optional[date] = Query(None, description="Start date inclusive (YYYY-MM-DD). Defaults to 30 days ago."),
    end_date: Optional[date] = Query(None, description="End date inclusive (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
) -> list[DailyFeaturesOut]:
    """Return daily feature rows for the requested date range, sorted oldest-first."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=29)

    return (
        db.query(DailyFeatures)
        .filter(DailyFeatures.date >= start_date, DailyFeatures.date <= end_date)
        .order_by(DailyFeatures.date)
        .all()
    )


@router.get("/weekly", response_model=list[WeeklySummaryOut], tags=["dashboard"])
def get_weekly(
    start_date: Optional[date] = Query(None, description="Earliest week_start inclusive (YYYY-MM-DD). Defaults to 12 weeks ago."),
    end_date: Optional[date] = Query(None, description="Latest week_start inclusive (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
) -> list[WeeklySummaryOut]:
    """Return weekly summary rows for the requested range, sorted oldest-first."""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(weeks=12)

    return (
        db.query(WeeklySummary)
        .filter(WeeklySummary.week_start >= start_date, WeeklySummary.week_start <= end_date)
        .order_by(WeeklySummary.week_start)
        .all()
    )
