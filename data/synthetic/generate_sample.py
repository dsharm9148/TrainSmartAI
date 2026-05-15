"""
Synthetic Apple Health data generator.

Generates 90 days of realistic health data for demo purposes.
Includes progressive fitness trends, post-workout fatigue effects,
and occasional anomaly days to make insights and correlations visible.

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

# Personal baselines (moderately active adult)
BASE_STEPS = 8500
BASE_RHR = 58.0
BASE_SLEEP_HRS = 7.2
WORKOUT_DAYS_PER_WEEK = 3

# Anomaly days — indices where something "bad" happens (travel, illness, etc.)
ANOMALY_DAY_INDICES = {12, 31, 47, 63, 78}


def fmt_date(dt: datetime) -> str:
    return dt.strftime(f"%Y-%m-%d %H:%M:%S {TZ_OFFSET}")


def rand_normal(mean: float, std: float, low: float, high: float) -> float:
    val = random.gauss(mean, std)
    return round(max(low, min(high, val)), 2)


def generate_days() -> list[dict]:
    """
    Generate per-day baseline values with realistic effects:
    - Progressive fitness trend: steps +15%, RHR -5 bpm over 90 days
    - Post-workout effect: next day RHR elevated, sleep slightly shorter
    - Anomaly days: very low activity, elevated HR, poor sleep
    """
    days = []
    prev_workout = False  # whether the previous day was a workout day

    for i in range(90):
        d = START_DATE + timedelta(days=i)
        weekday = d.weekday()  # 0=Mon, 6=Sun

        # Progressive fitness: linear improvement over 90 days
        progress = i / 89.0  # 0.0 → 1.0
        step_trend = BASE_STEPS * (1 + 0.15 * progress)
        rhr_trend = BASE_RHR - (5.0 * progress)

        is_anomaly = i in ANOMALY_DAY_INDICES

        if is_anomaly:
            # Bad day: low steps, elevated RHR, poor sleep
            steps = int(rand_normal(2500, 800, 500, 5000))
            rhr = rand_normal(rhr_trend + 6, 2, 50, 90)
            sleep_hrs = rand_normal(5.5, 0.8, 3.0, 7.0)
            avg_hr_mean = rhr + 15
            is_workout = False
        else:
            # Normal day: weekend slightly lower activity
            step_mean = step_trend if weekday < 5 else step_trend * 0.82
            sleep_mean = BASE_SLEEP_HRS if weekday < 5 else BASE_SLEEP_HRS + 0.45

            # Post-workout effect: elevated RHR, slightly shorter sleep
            rhr_bump = rand_normal(3.5, 1.0, 1.5, 6.0) if prev_workout else 0.0
            sleep_penalty = rand_normal(0.4, 0.15, 0.1, 0.8) if prev_workout else 0.0

            steps = int(rand_normal(step_mean, 2000, 1500, 22000))
            rhr = rand_normal(rhr_trend + rhr_bump, 3, 40, 85)
            sleep_hrs = rand_normal(sleep_mean - sleep_penalty, 1.0, 4.0, 10.5)
            avg_hr_mean = rhr + rand_normal(8, 3, 3, 20)
            is_workout = (not weekday == 6) and random.random() < (WORKOUT_DAYS_PER_WEEK / 7)

        days.append(
            {
                "date": d,
                "steps": steps,
                "rhr": rhr,
                "sleep_hrs": sleep_hrs,
                "avg_hr_mean": avg_hr_mean,
                "is_workout_day": is_workout,
                "is_anomaly": is_anomaly,
            }
        )
        prev_workout = is_workout

    return days


def build_xml(days: list[dict]) -> ET.Element:
    root = ET.Element("HealthData", locale="en_US")
    ET.SubElement(root, "ExportDate", value=fmt_date(END_DATE))

    for day in days:
        d = day["date"]

        # Steps — one record per day
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

        # Heart rate — 5–8 spot readings spread across waking hours
        num_hr = random.randint(5, 8)
        for _ in range(num_hr):
            hour = random.randint(7, 22)
            ts = d.replace(hour=hour, minute=random.randint(0, 59), second=0)
            hr_val = rand_normal(day["avg_hr_mean"], 10, 45, 140)
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

        # Resting heart rate — 1 per day
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

        # Sleep — bedtime ~10–11 pm, wake up based on duration
        bedtime_hour = random.randint(22, 23)
        sleep_start = d.replace(hour=bedtime_hour, minute=random.randint(0, 59), second=0)
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

        # Workout — on non-anomaly workout days
        if day["is_workout_day"] and not day["is_anomaly"]:
            activity = random.choice([
                "HKWorkoutActivityTypeRunning",
                "HKWorkoutActivityTypeCycling",
                "HKWorkoutActivityTypeWalking",
            ])
            duration_min = rand_normal(38, 12, 15, 90)
            calories = rand_normal(290, 80, 100, 600)
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
    print(f"Generating 90-day synthetic export ({START_DATE.date()} → END_DATE {END_DATE.date()})...")
    days = generate_days()
    root = build_xml(days)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(OUTPUT_PATH), encoding="utf-8", xml_declaration=True)

    workout_days = sum(1 for d in days if d["is_workout_day"])
    anomaly_days = sum(1 for d in days if d["is_anomaly"])
    print(f"  {len(root)} XML elements written to {OUTPUT_PATH}")
    print(f"  {workout_days} workout days, {anomaly_days} anomaly days (illness/travel)")
    print("  Fitness trend: steps +15%, resting HR -5 bpm over 90 days")
    print("Upload data/synthetic/demo_export.xml on the Upload page to demo without personal data.")


if __name__ == "__main__":
    main()
