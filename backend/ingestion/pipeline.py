"""
Ingestion pipeline orchestrator.

Ties together parse → clean → load into a single callable function.
This is what the FastAPI upload route will call — it handles the full
flow and returns a human-readable summary.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from backend.ingestion.cleaner import clean_records
from backend.ingestion.loader import get_record_count_by_type, load_records
from backend.ingestion.parser import parse_export


def run_ingestion(file_path: str | Path, db: Session) -> dict:
    """
    Run the full ingestion pipeline on an Apple Health export file.

    Steps:
        1. Parse — stream the XML, yield raw record dicts
        2. Clean — validate values, normalize types, filter bad data
        3. Load  — bulk upsert into health_records (idempotent)

    Returns a summary dict:
        parsed   — total records yielded by the parser
        cleaned  — records that passed validation
        inserted — newly inserted rows (0 if all already existed)
        filtered — records dropped by the cleaner
        by_type  — count of stored records broken down by type
    """
    file_path = Path(file_path)

    # Step 1: Parse
    raw_records = list(parse_export(file_path))

    # Step 2: Clean
    cleaned = clean_records(raw_records)

    # Step 3: Load
    inserted = load_records(db, cleaned)

    return {
        "parsed": len(raw_records),
        "cleaned": len(cleaned),
        "inserted": inserted,
        "filtered": len(raw_records) - len(cleaned),
        "by_type": get_record_count_by_type(db),
    }
