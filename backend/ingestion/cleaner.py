"""
Data cleaner for raw Apple Health records.

Handles all type-specific conversion and validation:
- Parses Apple's timestamp format into timezone-aware datetimes
- Converts string values to appropriate Python types
- Filters physiologically impossible values
- Converts sleep category values to duration in hours

Raw records come in with string timestamps and string values.
Clean records go out with datetime objects and typed values.
"""

from datetime import datetime
from typing import Optional

# Apple Health timestamp format: "2024-01-15 08:30:00 -0700"
APPLE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"

# Physiological bounds for filtering sensor errors
HR_MIN = 20.0
HR_MAX = 250.0
SLEEP_MAX_HOURS = 16.0
STEPS_MAX_PER_RECORD = 100_000  # per individual record, not per day

# Apple's sleep category values that represent actual sleep time
# We exclude InBed and Awake — those inflate duration without being real sleep
SLEEP_ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}

STEP_COUNT = "HKQuantityTypeIdentifierStepCount"
HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"


def clean_records(raw_records: list[dict]) -> list[dict]:
    """
    Clean and validate a list of raw parsed records.

    Invalid records are silently dropped — the caller gets back only
    records that are safe to store. Log the counts if you need to audit
    how many were filtered.
    """
    cleaned = []
    for raw in raw_records:
        record = _clean_record(raw)
        if record is not None:
            cleaned.append(record)
    return cleaned


def _clean_record(raw: dict) -> Optional[dict]:
    """
    Clean a single raw record. Returns None if the record should be dropped.

    The returned dict maps directly to the health_records table columns.
    """
    record_type = raw.get("record_type", "")

    start_date = _parse_date(raw.get("start_date"))
    end_date = _parse_date(raw.get("end_date"))

    if start_date is None or end_date is None:
        return None
    if end_date <= start_date:
        return None

    raw_value = raw.get("value")
    cleaned_value = None

    if record_type in (HEART_RATE, RESTING_HR):
        numeric = _safe_float(raw_value)
        if numeric is None or not (HR_MIN <= numeric <= HR_MAX):
            return None
        cleaned_value = numeric

    elif record_type == STEP_COUNT:
        numeric = _safe_float(raw_value)
        if numeric is None or numeric < 0 or numeric > STEPS_MAX_PER_RECORD:
            return None
        cleaned_value = int(numeric)

    elif record_type == SLEEP:
        # Value is a category string, not a number
        # We keep only actual sleep records and convert duration to hours
        if raw_value not in SLEEP_ASLEEP_VALUES:
            return None
        duration_hrs = (end_date - start_date).total_seconds() / 3600
        if duration_hrs <= 0 or duration_hrs > SLEEP_MAX_HOURS:
            return None
        cleaned_value = round(duration_hrs, 4)

    elif record_type.startswith("HKWorkoutActivityType"):
        numeric = _safe_float(raw_value)
        if numeric is not None and numeric < 0:
            return None
        cleaned_value = numeric

    else:
        # Unknown type — pass value through as float if possible
        cleaned_value = _safe_float(raw_value)

    return {
        "record_type": record_type,
        "source_name": raw.get("source_name"),
        "value": cleaned_value,
        "unit": raw.get("unit"),
        "start_date": start_date,
        "end_date": end_date,
    }


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse Apple Health date string to timezone-aware datetime."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), APPLE_DATE_FORMAT)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Convert a string to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
