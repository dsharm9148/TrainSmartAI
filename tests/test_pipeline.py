"""
End-to-end integration tests for the ingestion pipeline.

These tests exercise the full parse → clean → load flow against a real
Postgres instance. They require docker-compose postgres to be running.
"""

from pathlib import Path

from backend.ingestion.pipeline import run_ingestion

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"


def test_pipeline_runs_without_error(db_session):
    """Pipeline completes without raising an exception."""
    result = run_ingestion(FIXTURE_XML, db_session)
    assert result is not None


def test_pipeline_parsed_count(db_session):
    """Parser yields the expected number of records from the fixture.

    Fixture breakdown (25 total):
      5 step records
      8 heart rate records (including 1 invalid at 300 BPM)
      5 resting heart rate records
      5 sleep records (including 1 InBed)
      2 workout records
      -- body mass record is filtered by the parser itself (not in RECORD_TYPES)
    """
    result = run_ingestion(FIXTURE_XML, db_session)
    assert result["parsed"] == 25


def test_pipeline_cleaned_count(db_session):
    """Cleaner removes the 2 invalid records, leaving 23 clean records.

    Dropped:
      1 heart rate record (value=300, above HR_MAX of 250)
      1 sleep record (HKCategoryValueSleepAnalysisInBed)
    """
    result = run_ingestion(FIXTURE_XML, db_session)
    assert result["cleaned"] == 23
    assert result["filtered"] == 2


def test_pipeline_inserted_count(db_session):
    """First run inserts all 23 cleaned records."""
    result = run_ingestion(FIXTURE_XML, db_session)
    assert result["inserted"] == 23


def test_pipeline_is_idempotent(db_session):
    """Running the pipeline twice inserts 0 new records on the second run."""
    first = run_ingestion(FIXTURE_XML, db_session)
    second = run_ingestion(FIXTURE_XML, db_session)

    assert first["inserted"] == 23
    assert second["inserted"] == 0


def test_pipeline_by_type_breakdown(db_session):
    """Summary includes a per-type record count after ingestion."""
    result = run_ingestion(FIXTURE_XML, db_session)

    by_type = result["by_type"]
    assert by_type["HKQuantityTypeIdentifierStepCount"] == 5
    assert by_type["HKQuantityTypeIdentifierHeartRate"] == 7  # 8 parsed, 1 filtered
    assert by_type["HKQuantityTypeIdentifierRestingHeartRate"] == 5
    assert by_type["HKCategoryTypeIdentifierSleepAnalysis"] == 4  # 5 parsed, 1 filtered
    assert by_type["HKWorkoutActivityTypeRunning"] == 1
    assert by_type["HKWorkoutActivityTypeCycling"] == 1


def test_pipeline_summary_keys(db_session):
    """Result dict has all expected keys."""
    result = run_ingestion(FIXTURE_XML, db_session)
    assert set(result.keys()) == {"parsed", "cleaned", "inserted", "filtered", "by_type"}
