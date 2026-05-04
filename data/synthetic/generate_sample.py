"""
Synthetic Apple Health data generator.

Generates 90 days of realistic health data for demo purposes.
Run from the project root:

    python data/synthetic/generate_sample.py

Outputs to data/synthetic/demo_export.xml
"""

import random
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "demo_export.xml"
SEED = 42  # reproducible output
random.seed(SEED)

# 90 days ending yesterday
END_DATE = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
START_DATE = END_DATE - timedelta(days=90)
TZ_OFFSET = "-0500"
TZ = timezone(timedelta(hours=-5))

# Personal baselines (realistic for a moderately active adult)
BASE_STEPS = 8500
BASE_RHR = 58.0
BASE_SLEEP_HRS = 7.2
WORKOUT_DAYS_PER_WEEK = 3


def fmt_date(dt: datetime) -> str:
    return dt.strftime(f"%Y-%m-%d %H:%M:%S {TZ_OFFSET}")


def rand_normal(mean: float, std: float, low: float, high: float) -> float:
    val = random.gauss(mean, std)
    return round(max(low, min(high, val)), 2)


def generate_days() -> list[dict]:
    """Generate per-day baseline values for 90 days."""
    days = []
    for i in range(90):
        d = START_DATE + timedelta(days=i)
        weekday = d.weekday()  # 0=Mon, 6=Sun

        # Slightly lower activity on weekends
        step_mean = BASE_STEPS if weekday < 5 else BASE_STEPS * 0.85
        sleep_mean = BASE_SLEEP_HRS if weekday < 5 else BASE_SLEEP_HRS + 0.5

        days.append(
            {
                "date": d,
                "steps": int(rand_normal(step_mean, 2500, 1000, 25000)),
                "rhr": rand_normal(BASE_RHR, 4, 40, 90),
                "sleep_hrs": rand_normal(sleep_mean, 1.2, 3.0, 10.5),
                "is_workout_day": random.random() < (WORKOUT_DAYS_PER_WEEK / 7),
            }
        )
    return days


def build_xml(days: list[dict]) -> ET.Element:
    root = ET.Element("HealthData", locale="en_US")
    ET.SubElement(root, "ExportDate", value=fmt_date(END_DATE))

    for day in days:
        d = day["date"]

        # Steps (single record per day)
        ET.SubElement(
            root,
            "Record",
            type="HKQuantityTypeIdentifierStepCount",
            sourceName="iPhone",
            unit="count",
            value=str(day["steps"]),
            startDate=fmt_date(d.replace(hour=0, minute=0, second=0)),
            endDate=fmt_date(d.replace(hour=23, minute=59, second=59)),
        )

        # Heart rate (5-8 readings throughout the day)
        num_hr = random.randint(5, 8)
        for _ in range(num_hr):
            hour = random.randint(7, 22)
            minute = random.randint(0, 59)
            ts = d.replace(hour=hour, minute=minute, second=0)
            hr_val = rand_normal(65, 12, 45, 140)
            ET.SubElement(
                root,
                "Record",
                type="HKQuantityTypeIdentifierHeartRate",
                sourceName="Apple Watch",
                unit="count/min",
                value=str(round(hr_val)),
                startDate=fmt_date(ts),
                endDate=fmt_date(ts + timedelta(seconds=5)),
            )

        # Resting heart rate (1 per day)
        ET.SubElement(
            root,
            "Record",
            type="HKQuantityTypeIdentifierRestingHeartRate",
            sourceName="Apple Watch",
            unit="count/min",
            value=str(round(day["rhr"])),
            startDate=fmt_date(d.replace(hour=0, minute=0, second=0)),
            endDate=fmt_date(d.replace(hour=23, minute=59, second=59)),
        )

        # Sleep (bedtime ~10pm-midnight, wake up based on duration)
        bedtime_hour = random.randint(22, 23)
        bedtime_minute = random.randint(0, 59)
        sleep_start = d.replace(hour=bedtime_hour, minute=bedtime_minute, second=0)
        sleep_end = sleep_start + timedelta(hours=day["sleep_hrs"])
        ET.SubElement(
            root,
            "Record",
            type="HKCategoryTypeIdentifierSleepAnalysis",
            sourceName="iPhone",
            value="HKCategoryValueSleepAnalysisAsleep",
            startDate=fmt_date(sleep_start),
            endDate=fmt_date(sleep_end),
        )

        # Workout (on workout days)
        if day["is_workout_day"]:
            workout_types = [
                "HKWorkoutActivityTypeRunning",
                "HKWorkoutActivityTypeCycling",
                "HKWorkoutActivityTypeWalking",
            ]
            activity = random.choice(workout_types)
            duration_min = rand_normal(35, 12, 15, 90)
            calories = rand_normal(280, 80, 100, 600)
            wo_hour = random.choice([6, 7, 12, 17, 18])
            wo_start = d.replace(hour=wo_hour, minute=0, second=0)
            wo_end = wo_start + timedelta(minutes=duration_min)
            ET.SubElement(
                root,
                "Workout",
                activityType=activity,
                duration=str(round(duration_min, 1)),
                durationUnit="min",
                totalEnergyBurned=str(round(calories)),
                totalEnergyBurnedUnit="kcal",
                sourceName="Apple Watch",
                startDate=fmt_date(wo_start),
                endDate=fmt_date(wo_end),
            )

    return root


def main():
    print(f"Generating 90-day synthetic export ({START_DATE.date()} → {END_DATE.date()})...")
    days = generate_days()
    root = build_xml(days)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(OUTPUT_PATH), encoding="utf-8", xml_declaration=True)

    total_records = len(root)
    print(f"Written {total_records} elements to {OUTPUT_PATH}")
    print("Use this file on the Upload page to demo without personal data.")


if __name__ == "__main__":
    main()
