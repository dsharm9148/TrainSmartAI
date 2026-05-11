from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    parsed: int
    cleaned: int
    inserted: int
    filtered: int
    days_computed: int
    message: str
    by_type: dict[str, int]


class DailyFeaturesOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    steps: Optional[int] = None
    avg_heart_rate: Optional[float] = None
    resting_heart_rate: Optional[float] = None
    sleep_duration_hrs: Optional[float] = None
    sleep_start: Optional[datetime] = None
    sleep_end: Optional[datetime] = None
    workout_count: Optional[int] = None
    workout_minutes: Optional[float] = None
    workout_calories: Optional[float] = None
    active_energy_kcal: Optional[float] = None


class WeeklySummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    week_start: date
    week_end: date
    avg_daily_steps: Optional[float] = None
    avg_sleep_hrs: Optional[float] = None
    sleep_consistency_score: Optional[float] = None
    avg_resting_hr: Optional[float] = None
    total_workout_minutes: Optional[float] = None
    workout_days: Optional[int] = None
    avg_readiness_score: Optional[float] = None
