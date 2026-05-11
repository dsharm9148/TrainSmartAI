"""
Tests for Day 6 FastAPI routes: POST /upload, GET /daily, GET /weekly.
"""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.db.models import DailyFeatures, WeeklySummary
from backend.db.session import get_db


def _db_override(session):
    def _dep():
        yield session

    return _dep


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_db] = _db_override(db_session)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_daily(db_session):
    rows = [
        DailyFeatures(
            date=date(2024, 1, 2),
            steps=10234,
            avg_heart_rate=70.0,
            resting_heart_rate=60.0,
            sleep_duration_hrs=8.0,
            workout_count=1,
            workout_minutes=30.5,
        ),
        DailyFeatures(
            date=date(2024, 1, 3),
            steps=6789,
            avg_heart_rate=75.0,
            resting_heart_rate=57.0,
            sleep_duration_hrs=7.0,
        ),
    ]
    db_session.add_all(rows)
    db_session.flush()
    return rows


@pytest.fixture
def seeded_weekly(db_session):
    row = WeeklySummary(
        week_start=date(2024, 1, 1),
        week_end=date(2024, 1, 7),
        avg_daily_steps=8500.0,
        avg_sleep_hrs=7.5,
        total_workout_minutes=75.7,
        workout_days=2,
    )
    db_session.add(row)
    db_session.flush()
    return row


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_valid_xml_returns_200(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        with open(xml_path, "rb") as f:
            resp = client.post("/upload", files={"file": ("export.xml", f, "application/xml")})
        assert resp.status_code == 200

    def test_response_shape(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        with open(xml_path, "rb") as f:
            body = client.post("/upload", files={"file": ("export.xml", f, "application/xml")}).json()
        for key in ("parsed", "cleaned", "inserted", "filtered", "days_computed", "message", "by_type"):
            assert key in body, f"missing key: {key}"

    def test_parsed_count_positive(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        with open(xml_path, "rb") as f:
            body = client.post("/upload", files={"file": ("export.xml", f, "application/xml")}).json()
        assert body["parsed"] > 0

    def test_days_computed_non_negative(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        with open(xml_path, "rb") as f:
            body = client.post("/upload", files={"file": ("export.xml", f, "application/xml")}).json()
        assert body["days_computed"] >= 0

    def test_rejects_non_xml_file(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")},
        )
        assert resp.status_code == 400
        assert "xml" in resp.json()["detail"].lower()

    def test_idempotent_second_upload_inserts_zero(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        for _ in range(2):
            with open(xml_path, "rb") as f:
                resp = client.post("/upload", files={"file": ("export.xml", f, "application/xml")})
            assert resp.status_code == 200
        assert resp.json()["inserted"] == 0

    def test_by_type_is_dict(self, client, fixtures_dir):
        xml_path = fixtures_dir / "sample_export.xml"
        with open(xml_path, "rb") as f:
            body = client.post("/upload", files={"file": ("export.xml", f, "application/xml")}).json()
        assert isinstance(body["by_type"], dict)


# ---------------------------------------------------------------------------
# GET /daily
# ---------------------------------------------------------------------------

class TestGetDaily:
    def test_returns_200(self, client):
        resp = client.get("/daily")
        assert resp.status_code == 200

    def test_default_returns_list(self, client):
        assert isinstance(client.get("/daily").json(), list)

    def test_rows_in_range(self, client, seeded_daily):
        resp = client.get("/daily", params={"start_date": "2024-01-01", "end_date": "2024-01-07"})
        rows = resp.json()
        assert len(rows) == 2

    def test_sorted_oldest_first(self, client, seeded_daily):
        rows = client.get("/daily", params={"start_date": "2024-01-01", "end_date": "2024-01-07"}).json()
        dates = [r["date"] for r in rows]
        assert dates == sorted(dates)

    def test_correct_field_values(self, client, seeded_daily):
        rows = client.get("/daily", params={"start_date": "2024-01-01", "end_date": "2024-01-07"}).json()
        jan2 = next(r for r in rows if r["date"] == "2024-01-02")
        assert jan2["steps"] == 10234
        assert jan2["workout_count"] == 1

    def test_null_fields_returned_as_none(self, client, seeded_daily):
        rows = client.get("/daily", params={"start_date": "2024-01-01", "end_date": "2024-01-07"}).json()
        jan3 = next(r for r in rows if r["date"] == "2024-01-03")
        assert jan3["workout_count"] is None
        assert jan3["workout_minutes"] is None

    def test_empty_range_returns_empty_list(self, client):
        resp = client.get("/daily", params={"start_date": "2020-01-01", "end_date": "2020-01-31"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_excludes_rows_outside_range(self, client, seeded_daily):
        rows = client.get("/daily", params={"start_date": "2024-01-03", "end_date": "2024-01-07"}).json()
        assert all(r["date"] >= "2024-01-03" for r in rows)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# GET /weekly
# ---------------------------------------------------------------------------

class TestGetWeekly:
    def test_returns_200(self, client):
        assert client.get("/weekly").status_code == 200

    def test_default_returns_list(self, client):
        assert isinstance(client.get("/weekly").json(), list)

    def test_row_in_range(self, client, seeded_weekly):
        rows = client.get("/weekly", params={"start_date": "2023-12-25", "end_date": "2024-01-07"}).json()
        assert len(rows) == 1
        assert rows[0]["week_start"] == "2024-01-01"
        assert rows[0]["week_end"] == "2024-01-07"

    def test_correct_field_values(self, client, seeded_weekly):
        rows = client.get("/weekly", params={"start_date": "2023-12-25", "end_date": "2024-01-07"}).json()
        assert rows[0]["avg_daily_steps"] == 8500.0
        assert rows[0]["workout_days"] == 2

    def test_null_optional_fields(self, client, seeded_weekly):
        rows = client.get("/weekly", params={"start_date": "2023-12-25", "end_date": "2024-01-07"}).json()
        assert rows[0]["avg_readiness_score"] is None
        assert rows[0]["sleep_consistency_score"] is None

    def test_empty_range_returns_empty_list(self, client):
        resp = client.get("/weekly", params={"start_date": "2020-01-01", "end_date": "2020-12-31"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_excludes_rows_outside_range(self, client, seeded_weekly):
        rows = client.get("/weekly", params={"start_date": "2024-01-08", "end_date": "2024-12-31"}).json()
        assert len(rows) == 0
