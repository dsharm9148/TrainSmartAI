"""
Apple Health XML parser.

Uses iterparse for streaming — handles multi-GB exports without loading
the entire file into memory. Filters to the four metric types this app cares
about: steps, heart rate, sleep, and workouts.

Raw records are returned as plain dicts with string timestamps. The cleaner
module handles type conversion and validation.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator

# Apple Health record type identifiers
STEP_COUNT = "HKQuantityTypeIdentifierStepCount"
HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"

# Only these types are extracted from <Record> elements
RECORD_TYPES = {STEP_COUNT, HEART_RATE, RESTING_HR, SLEEP}


def parse_export(file_path: str | Path) -> Generator[dict, None, None]:
    """
    Stream-parse an Apple Health export.xml and yield raw record dicts.

    Uses iterparse so memory usage stays flat regardless of file size.
    Each yielded dict has keys:
        record_type, source_name, value (raw string), unit, start_date, end_date

    Yields <Record> elements for the 4 types we care about, plus all
    <Workout> elements.
    """
    file_path = Path(file_path)

    for _event, elem in ET.iterparse(str(file_path), events=("end",)):
        if elem.tag == "Record":
            record_type = elem.get("type", "")
            if record_type in RECORD_TYPES:
                raw = _parse_record_element(elem, record_type)
                if raw is not None:
                    yield raw

        elif elem.tag == "Workout":
            raw = _parse_workout_element(elem)
            if raw is not None:
                yield raw

        # Free the element from memory — critical for large files
        elem.clear()


def _parse_record_element(elem: ET.Element, record_type: str) -> dict | None:
    """
    Parse a <Record> element into a raw dict.

    Value is returned as a raw string — the cleaner handles conversion.
    Sleep records have a category string (e.g. HKCategoryValueSleepAnalysisAsleep)
    as their value, not a number.
    """
    start_str = elem.get("startDate")
    end_str = elem.get("endDate")

    if not start_str or not end_str:
        return None

    return {
        "record_type": record_type,
        "source_name": elem.get("sourceName"),
        "value": elem.get("value"),  # raw string, cleaner converts
        "unit": elem.get("unit"),
        "start_date": start_str,
        "end_date": end_str,
    }


def _parse_workout_element(elem: ET.Element) -> dict | None:
    """
    Parse a <Workout> element into a raw dict.

    Workouts use different attributes than Record elements:
    - activityType instead of type
    - duration + durationUnit instead of value + unit
    """
    start_str = elem.get("startDate")
    end_str = elem.get("endDate")

    if not start_str or not end_str:
        return None

    return {
        "record_type": elem.get("activityType", "HKWorkoutActivityTypeOther"),
        "source_name": elem.get("sourceName"),
        "value": elem.get("duration"),  # raw string duration
        "unit": elem.get("durationUnit", "min"),
        "start_date": start_str,
        "end_date": end_str,
    }
