"""
Tests for the ingestion pipeline: parser → cleaner → loader.

Parser and cleaner tests are pure Python — no database required.
Loader tests require a running Postgres instance.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.ingestion.cleaner import clean_records, _parse_date, _safe_float
from backend.ingestion.parser import (
    RECORD_TYPES,
    SLEEP,
    STEP_COUNT,
    HEART_RATE,
    RESTING_HR,
    parse_export,
)

FIXTURE_XML = Path(__file__).parent / "fixtures" / "sample_export.xml"


# ─── Parser tests ─────────────────────────────────────────────────────────────

def test_parser_returns_records():
    """Parser yields at least one record from the sample fixture."""
    records = list(parse_export(FIXTURE_XML))
    assert len(records) > 0


def test_parser_yields_all_expected_types():
    """All four relevant record types appear in the parsed output."""
    records = list(parse_export(FIXTURE_XML))
    types_found = {r["record_type"] for r in records}
    assert STEP_COUNT in types_found
    assert HEART_RATE in types_found
    assert RESTING_HR in types_found
    assert SLEEP in types_found


def test_parser_includes_workouts():
    """Workout elements are parsed and included in output."""
    records = list(parse_export(FIXTURE_XML))
    workout_types = {r["record_type"] for r in records if r["record_type"].startswith("HKWorkoutActivityType")}
    assert len(workout_types) > 0
    assert "HKWorkoutActivityTypeRunning" in workout_types


def test_parser_filters_irrelevant_types():
    """Record types not in our set of interest are excluded."""
    records = list(parse_export(FIXTURE_XML))
    types = {r["record_type"] for r in records}
    assert "HKQuantityTypeIdentifierBodyMass" not in types


def test_parser_record_has_expected_keys():
    """Each parsed record has the required keys."""
    records = list(parse_export(FIXTURE_XML))
    required_keys = {"record_type", "source_name", "value", "unit", "start_date", "end_date"}
    for record in records:
        assert required_keys.issubset(record.keys()), f"Missing keys in: {record}"


def test_parser_timestamps_are_strings():
    """Parser returns raw string timestamps — not datetimes."""
    records = list(parse_export(FIXTURE_XML))
    for record in records:
        assert isinstance(record["start_date"], str)
        assert isinstance(record["end_date"], str)


def test_parser_step_count():
    """Parser yields 5 step records from the fixture."""
    records = list(parse_export(FIXTURE_XML))
    steps = [r for r in records if r["record_type"] == STEP_COUNT]
    assert len(steps) == 5


def test_parser_includes_inbed_sleep():
    """Parser passes through InBed sleep records — cleaner is responsible for filtering them."""
    records = list(parse_export(FIXTURE_XML))
    sleep_records = [r for r in records if r["record_type"] == SLEEP]
    # Fixture has 5 sleep records total (4 Asleep + 1 InBed)
    assert len(sleep_records) == 5


# ─── Cleaner tests ────────────────────────────────────────────────────────────

def test_cleaner_drops_invalid_hr():
    """Heart rate values outside physiological range are dropped."""
    raw = [
        {
            "record_type": HEART_RATE,
            "source_name": "Apple Watch",
            "value": "300",  # above HR_MAX of 250
            "unit": "count/min",
            "start_date": "2024-01-01 08:00:00 -0500",
            "end_date": "2024-01-01 08:00:05 -0500",
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 0


def test_cleaner_keeps_valid_hr():
    """Valid heart rate values are kept and converted to float."""
    raw = [
        {
            "record_type": HEART_RATE,
            "source_name": "Apple Watch",
            "value": "72",
            "unit": "count/min",
            "start_date": "2024-01-01 08:00:00 -0500",
            "end_date": "2024-01-01 08:00:05 -0500",
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["value"] == 72.0


def test_cleaner_drops_inbed_sleep():
    """InBed sleep category is dropped — only Asleep records are kept."""
    raw = [
        {
            "record_type": SLEEP,
            "source_name": "iPhone",
            "value": "HKCategoryValueSleepAnalysisInBed",
            "unit": None,
            "start_date": "2024-01-01 22:30:00 -0500",
            "end_date": "2024-01-02 07:30:00 -0500",
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 0


def test_cleaner_converts_sleep_to_hours():
    """Sleep duration is calculated from start/end timestamps and stored in hours."""
    raw = [
        {
            "record_type": SLEEP,
            "source_name": "iPhone",
            "value": "HKCategoryValueSleepAnalysisAsleep",
            "unit": None,
            "start_date": "2024-01-01 23:00:00 -0500",
            "end_date": "2024-01-02 07:00:00 -0500",  # 8 hours
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["value"] == pytest.approx(8.0, rel=0.01)


def test_cleaner_converts_steps_to_int():
    """Step values are cast to int."""
    raw = [
        {
            "record_type": STEP_COUNT,
            "source_name": "iPhone",
            "value": "8543.0",
            "unit": "count",
            "start_date": "2024-01-01 00:00:00 -0500",
            "end_date": "2024-01-01 23:59:59 -0500",
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 1
    assert isinstance(cleaned[0]["value"], int)
    assert cleaned[0]["value"] == 8543


def test_cleaner_outputs_datetime_objects():
    """Cleaned records have datetime objects, not strings."""
    raw = [
        {
            "record_type": STEP_COUNT,
            "source_name": "iPhone",
            "value": "5000",
            "unit": "count",
            "start_date": "2024-01-01 00:00:00 -0500",
            "end_date": "2024-01-01 23:59:59 -0500",
        }
    ]
    cleaned = clean_records(raw)
    assert len(cleaned) == 1
    assert isinstance(cleaned[0]["start_date"], datetime)
    assert isinstance(cleaned[0]["end_date"], datetime)
    assert cleaned[0]["start_date"].tzinfo is not None  # must be timezone-aware


def test_cleaner_on_full_fixture():
    """End-to-end: parse + clean the full fixture gives expected record count."""
    raw = list(parse_export(FIXTURE_XML))
    cleaned = clean_records(raw)

    # Fixture has:
    #   5 steps (valid)
    #   7 HR (valid) + 1 HR invalid (300 BPM) → 7 kept
    #   5 RHR (valid)
    #   4 Asleep sleep + 1 InBed → 4 kept
    #   2 workouts (valid)
    # Total: 5 + 7 + 5 + 4 + 2 = 23
    assert len(cleaned) == 23


def test_parse_date_handles_apple_format():
    """Date parser handles Apple's timestamp format correctly."""
    dt = _parse_date("2024-01-15 08:30:00 -0700")
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 1
    assert dt.day == 15
    assert dt.tzinfo is not None


def test_parse_date_returns_none_on_bad_input():
    """Date parser returns None for invalid or missing input."""
    assert _parse_date(None) is None
    assert _parse_date("") is None
    assert _parse_date("not a date") is None


def test_safe_float_handles_none():
    assert _safe_float(None) is None


def test_safe_float_handles_invalid_string():
    assert _safe_float("abc") is None


def test_safe_float_converts_valid_string():
    assert _safe_float("72.5") == 72.5


# ─── Loader tests (require running Postgres) ──────────────────────────────────

def test_loader_inserts_records(db_session):
    """Cleaned records from the fixture are successfully inserted."""
    from backend.ingestion.loader import load_records, get_record_count

    raw = list(parse_export(FIXTURE_XML))
    cleaned = clean_records(raw)

    inserted = load_records(db_session, cleaned)
    assert inserted == 23
    assert get_record_count(db_session) >= 23


def test_loader_is_idempotent(db_session):
    """Running the loader twice does not create duplicate records."""
    from backend.ingestion.loader import load_records, get_record_count

    raw = list(parse_export(FIXTURE_XML))
    cleaned = clean_records(raw)

    first_run = load_records(db_session, cleaned)
    count_after_first = get_record_count(db_session)

    second_run = load_records(db_session, cleaned)
    count_after_second = get_record_count(db_session)

    assert second_run == 0  # nothing new inserted
    assert count_after_first == count_after_second  # count unchanged
