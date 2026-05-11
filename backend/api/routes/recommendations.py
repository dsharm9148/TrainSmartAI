from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.schemas import RecommendationOut
from backend.db.models import Recommendation
from backend.db.session import get_db
from backend.recommendations.engine import LOOKBACK_DAYS, generate_recommendations

router = APIRouter()


@router.post("/recommendations/generate", tags=["recommendations"])
def trigger_recommendations(
    for_date: Optional[date] = Query(None, description="As-of date (YYYY-MM-DD). Defaults to today."),
    lookback_days: int = Query(LOOKBACK_DAYS, ge=5, le=90),
    db: Session = Depends(get_db),
) -> dict:
    """Generate rule-based recommendations from recent daily features and readiness scores."""
    count = generate_recommendations(db, for_date, lookback_days)
    return {"recommendations_generated": count, "message": f"Generated {count} recommendations."}


@router.get("/recommendations", response_model=list[RecommendationOut], tags=["recommendations"])
def get_recommendations(
    for_date: Optional[date] = Query(None, description="Filter by date (YYYY-MM-DD)."),
    category: Optional[str] = Query(None, description="Filter by category: recovery, workout, sleep, habit."),
    db: Session = Depends(get_db),
) -> list[RecommendationOut]:
    """Return stored recommendations, sorted by priority then date descending."""
    q = db.query(Recommendation)
    if for_date:
        q = q.filter(Recommendation.date == for_date)
    if category:
        q = q.filter(Recommendation.category == category)
    return q.order_by(Recommendation.date.desc(), Recommendation.priority).all()
