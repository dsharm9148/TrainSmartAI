"""
Day-type clustering endpoints — read assignments or trigger a refit.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.schemas import ClusterAssignmentOut
from backend.clustering.kmeans import compute_clusters
from backend.db.models import ClusterAssignment
from backend.db.session import get_db

router = APIRouter()


@router.post("/clusters/recompute", tags=["clusters"])
def trigger_recompute(
    n_clusters: int = Query(4, ge=2, le=8),
    db: Session = Depends(get_db),
) -> dict:
    """Re-fit K-means across all daily features and upsert cluster assignments."""
    count = compute_clusters(db, n_clusters=n_clusters)
    return {"days_clustered": count, "message": f"Clustered {count} days into {n_clusters} groups."}


@router.get("/clusters", response_model=list[ClusterAssignmentOut], tags=["clusters"])
def list_clusters(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
) -> list[ClusterAssignmentOut]:
    """List cluster assignments, optionally filtered by date range."""
    q = db.query(ClusterAssignment)
    if start_date:
        q = q.filter(ClusterAssignment.date >= start_date)
    if end_date:
        q = q.filter(ClusterAssignment.date <= end_date)
    return q.order_by(ClusterAssignment.date).all()
