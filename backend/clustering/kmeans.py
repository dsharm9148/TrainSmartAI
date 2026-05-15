"""
K-means day-type clustering.

Fits clusters across four normalized daily features (steps, sleep hours,
resting HR, workout minutes), then auto-labels each centroid with a
human-readable archetype like "training day" or "recovery day".

Public API:
  compute_clusters(db, n_clusters=4, random_state=42) -> int
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from backend.db.models import ClusterAssignment, DailyFeatures

FEATURES = ["steps", "sleep_duration_hrs", "resting_heart_rate", "workout_minutes"]


def compute_clusters(
    db: Session,
    n_clusters: int = 4,
    random_state: int = 42,
) -> int:
    """
    Fit K-means over recent daily features and upsert one ClusterAssignment per day.

    Days with too many missing features are skipped. Returns the number of
    days successfully clustered.
    """
    rows = db.query(DailyFeatures).order_by(DailyFeatures.date).all()
    if not rows:
        return 0

    df = pd.DataFrame([
        {
            "date": r.date,
            "steps": r.steps,
            "sleep_duration_hrs": r.sleep_duration_hrs,
            "resting_heart_rate": r.resting_heart_rate,
            "workout_minutes": r.workout_minutes or 0.0,
        }
        for r in rows
    ])

    # Require at least 3 of 4 features present per day
    df = df.dropna(subset=FEATURES, thresh=3).copy()
    if df.empty:
        return 0

    # Median-impute remaining missing values per feature
    for c in FEATURES:
        df[c] = df[c].fillna(df[c].median())

    # Drop any day still incomplete (e.g. column had no median because all NaN)
    df = df.dropna(subset=FEATURES)
    if len(df) < n_clusters:
        n_clusters = max(2, len(df))

    X = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = km.fit_predict(Xz)

    labels = _label_centroids(km.cluster_centers_)

    # Upsert each day's assignment
    for d, cid in zip(df["date"], cluster_ids):
        existing = (
            db.query(ClusterAssignment)
            .filter(ClusterAssignment.date == d)
            .first()
        )
        if existing is None:
            existing = ClusterAssignment(date=d)
            db.add(existing)
        existing.cluster_id = int(cid)
        existing.cluster_label = labels[int(cid)]

    db.commit()
    return len(df)


def _label_centroids(centers_z: np.ndarray) -> list[str]:
    """
    Assign a human-readable label to each centroid.

    Features (z-scored) in order: steps, sleep, resting HR, workout minutes.
    Labels are picked from the dominant z-score: whichever feature deviates
    most from the mean (positively or negatively) drives the descriptor.
    """
    labels: list[str] = []
    for c in centers_z:
        steps_z, sleep_z, hr_z, workout_z = c

        if workout_z > 0.6:
            labels.append("hard training day")
        elif steps_z > 0.6 and workout_z < 0.2:
            labels.append("active day")
        elif sleep_z > 0.6 and workout_z < 0.2:
            labels.append("deep recovery day")
        elif hr_z > 0.6 or sleep_z < -0.6:
            labels.append("stressed / under-recovered day")
        elif steps_z < -0.6:
            labels.append("sedentary day")
        else:
            labels.append("balanced day")
    return _disambiguate(labels)


def _disambiguate(labels: list[str]) -> list[str]:
    """Append #2, #3, ... when the same label repeats across multiple centroids."""
    seen: dict[str, int] = {}
    out = []
    for lbl in labels:
        seen[lbl] = seen.get(lbl, 0) + 1
        out.append(lbl if seen[lbl] == 1 else f"{lbl} #{seen[lbl]}")
    return out
