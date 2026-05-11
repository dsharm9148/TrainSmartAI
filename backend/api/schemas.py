from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    parsed: int
    cleaned: int
    inserted: int
    filtered: int
    days_computed: int
    weeks_computed: int
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


class ReadinessOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    score: float
    sleep_score: Optional[float] = None
    hr_score: Optional[float] = None
    load_score: Optional[float] = None
    consistency_score: Optional[float] = None
    explanation: Optional[str] = None


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    category: Optional[str] = None
    priority: Optional[int] = None
    text: str
    reasoning: Optional[str] = None


class InsightOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    generated_for: date
    insight_type: Optional[str] = None
    text: str
    created_at: datetime


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[UUID] = None


class ChatResponse(BaseModel):
    session_id: UUID
    response: str


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ChatSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID
    started_at: datetime
    last_active: datetime


class ClusterAssignmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: date
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None


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
