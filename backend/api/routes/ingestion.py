import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.api.schemas import UploadResponse
from backend.db.session import get_db
from backend.ingestion.pipeline import run_ingestion
from backend.preprocessing.daily_features import compute_daily_features
from backend.analytics.insights import generate_insights
from backend.analytics.weekly import compute_weekly_summaries
from backend.recommendations.engine import generate_recommendations
from backend.scoring.readiness import compute_readiness_scores

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, tags=["ingestion"])
async def upload_export(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Upload an Apple Health export.xml and ingest all health records.

    Runs parse → clean → load → daily feature aggregation.
    Safe to re-upload the same file: duplicates are silently skipped.
    """
    if not file.filename or not file.filename.lower().endswith(".xml"):
        raise HTTPException(status_code=400, detail="File must be an Apple Health XML export (.xml)")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        summary = run_ingestion(tmp_path, db)
        days_computed = compute_daily_features(db)
        compute_readiness_scores(db)
        weeks_computed = compute_weekly_summaries(db)
        generate_insights(db)
        generate_recommendations(db)
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(
        **summary,
        days_computed=days_computed,
        weeks_computed=weeks_computed,
        message=f"Ingested {summary['inserted']} new records across {days_computed} days.",
    )
