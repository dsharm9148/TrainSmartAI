"""
Database loader for cleaned health records.

Bulk-upserts records into the health_records table in batches.
Uses ON CONFLICT DO NOTHING so re-running ingestion on the same
export is always safe — existing records are skipped, not duplicated.

Batch inserts are used to avoid hitting SQL statement size limits
on very large exports.
"""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.db.models import HealthRecord

BATCH_SIZE = 1000  # rows per INSERT statement


def load_records(db: Session, records: list[dict]) -> int:
    """
    Upsert cleaned records into health_records in batches.

    Returns the number of newly inserted rows. Rows that already exist
    (based on the unique constraint) are silently skipped.
    """
    if not records:
        return 0

    total_inserted = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        stmt = insert(HealthRecord).values(batch)
        stmt = stmt.on_conflict_do_nothing(constraint="uq_health_record")
        result = db.execute(stmt)
        total_inserted += result.rowcount or 0

    db.commit()
    return total_inserted


def get_record_count(db: Session) -> int:
    """Return total number of raw records currently stored."""
    return db.query(HealthRecord).count()


def get_record_count_by_type(db: Session) -> dict[str, int]:
    """Return count of records grouped by type — useful for upload summary."""
    from sqlalchemy import func

    rows = (
        db.query(HealthRecord.record_type, func.count(HealthRecord.id))
        .group_by(HealthRecord.record_type)
        .all()
    )
    return {record_type: count for record_type, count in rows}
