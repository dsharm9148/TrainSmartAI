"""
Tests for backend/clustering/kmeans.py.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from backend.clustering.kmeans import (
    _disambiguate,
    _label_centroids,
    compute_clusters,
)
from backend.db.models import ClusterAssignment, DailyFeatures

D1 = date(2025, 1, 6)


def _day(db, d: date, *, steps=8000, sleep=7.0, rhr=55.0, workout_min=0.0) -> DailyFeatures:
    row = DailyFeatures(
        date=d,
        steps=steps,
        sleep_duration_hrs=sleep,
        resting_heart_rate=rhr,
        workout_minutes=workout_min,
        workout_count=1 if workout_min > 0 else 0,
    )
    db.add(row)
    return row


# ─── Label helpers ─────────────────────────────────────────────────────────


class TestLabelCentroids:
    def test_high_workout_labeled_training(self):
        # steps_z, sleep_z, hr_z, workout_z
        centers = np.array([[0.0, 0.0, 0.0, 1.5]])
        assert _label_centroids(centers)[0] == "hard training day"

    def test_high_steps_low_workout_labeled_active(self):
        centers = np.array([[1.2, 0.0, 0.0, 0.0]])
        assert _label_centroids(centers)[0] == "active day"

    def test_high_sleep_low_workout_labeled_recovery(self):
        centers = np.array([[0.0, 1.5, 0.0, 0.0]])
        assert _label_centroids(centers)[0] == "deep recovery day"

    def test_high_hr_labeled_stressed(self):
        centers = np.array([[0.0, 0.0, 1.5, 0.0]])
        assert _label_centroids(centers)[0] == "stressed / under-recovered day"

    def test_low_steps_labeled_sedentary(self):
        centers = np.array([[-1.2, 0.0, 0.0, 0.0]])
        assert _label_centroids(centers)[0] == "sedentary day"

    def test_average_labeled_balanced(self):
        centers = np.array([[0.0, 0.0, 0.0, 0.0]])
        assert _label_centroids(centers)[0] == "balanced day"


class TestDisambiguate:
    def test_unique_labels_unchanged(self):
        assert _disambiguate(["a", "b"]) == ["a", "b"]

    def test_duplicates_get_suffix(self):
        assert _disambiguate(["a", "a", "a"]) == ["a", "a #2", "a #3"]


# ─── Integration tests ──────────────────────────────────────────────────────


class TestComputeClusters:
    def test_empty_db_returns_zero(self, db_session):
        assert compute_clusters(db_session) == 0

    def test_returns_count_of_clustered_days(self, db_session):
        # 8 distinct days, varied profiles
        for i in range(8):
            _day(
                db_session,
                D1 + timedelta(days=i),
                steps=4000 + i * 1000,
                sleep=6.5 + (i % 3) * 0.5,
                rhr=52 + i % 4,
                workout_min=(40.0 if i % 3 == 0 else 0.0),
            )
        db_session.flush()
        count = compute_clusters(db_session, n_clusters=3)
        assert count == 8

    def test_assignment_rows_written(self, db_session):
        for i in range(6):
            _day(db_session, D1 + timedelta(days=i), steps=5000 + i * 500)
        db_session.flush()
        compute_clusters(db_session, n_clusters=3)
        rows = db_session.query(ClusterAssignment).all()
        assert len(rows) == 6
        for r in rows:
            assert r.cluster_id is not None
            assert r.cluster_label is not None
            assert len(r.cluster_label) > 0

    def test_idempotent(self, db_session):
        for i in range(5):
            _day(db_session, D1 + timedelta(days=i))
        db_session.flush()
        c1 = compute_clusters(db_session, n_clusters=2)
        c2 = compute_clusters(db_session, n_clusters=2)
        assert c1 == c2 == 5
        assert db_session.query(ClusterAssignment).count() == 5

    def test_skips_days_with_too_many_missing_features(self, db_session):
        # 3 good days
        for i in range(3):
            _day(db_session, D1 + timedelta(days=i))
        # 1 day with only steps — should be skipped (only 1 of 4 features non-null)
        db_session.add(DailyFeatures(date=D1 + timedelta(days=3), steps=8000))
        db_session.flush()
        count = compute_clusters(db_session, n_clusters=2)
        assert count == 3

    def test_reduces_k_when_data_smaller_than_n_clusters(self, db_session):
        for i in range(3):
            _day(db_session, D1 + timedelta(days=i))
        db_session.flush()
        # Request 5 clusters but only 3 rows → should still succeed
        count = compute_clusters(db_session, n_clusters=5)
        assert count == 3
        rows = db_session.query(ClusterAssignment).all()
        ids = {r.cluster_id for r in rows}
        assert len(ids) <= 3

    def test_deterministic_with_random_state(self, db_session):
        for i in range(10):
            _day(db_session, D1 + timedelta(days=i), steps=4000 + i * 800, workout_min=30.0 * (i % 2))
        db_session.flush()
        compute_clusters(db_session, n_clusters=3, random_state=42)
        first = {r.date: r.cluster_id for r in db_session.query(ClusterAssignment).all()}
        compute_clusters(db_session, n_clusters=3, random_state=42)
        second = {r.date: r.cluster_id for r in db_session.query(ClusterAssignment).all()}
        assert first == second
