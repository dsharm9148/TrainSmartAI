"""
Tests for the RAG layer: document builders, indexer, and chain.

OpenAI API calls are never made — a FakeEmbeddings class stands in for
real embeddings, and the LLM is replaced with a MagicMock where needed.
Chroma is pointed at a pytest tmp_path so tests don't write to ./chroma_db.
"""
from __future__ import annotations

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from backend.db.models import (
    DailyFeatures,
    Insight,
    ReadinessScore,
    Recommendation,
    WeeklySummary,
)
from backend.rag.documents import (
    build_daily_doc,
    build_insight_doc,
    build_readiness_doc,
    build_recommendation_doc,
    build_weekly_doc,
    load_all_documents,
)

D1 = date(2025, 1, 6)


# ─── Fake embeddings (no API calls) ─────────────────────────────────────────


class _FakeEmbeddings(Embeddings):
    """Deterministic unit-length embeddings — avoids any OpenAI call."""

    dim = 16

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _redirect_chroma(monkeypatch, tmp_path):
    """Point chroma_persist_dir at a temp dir for the duration of the test."""
    import backend.config as cfg
    monkeypatch.setattr(cfg.settings, "chroma_persist_dir", str(tmp_path))


# ─── Document builders ───────────────────────────────────────────────────────


class TestBuildDailyDoc:
    def test_contains_date(self):
        row = DailyFeatures(date=D1, steps=8000)
        assert "2025-01-06" in build_daily_doc(row).page_content

    def test_metadata_type(self):
        assert build_daily_doc(DailyFeatures(date=D1)).metadata["type"] == "daily"

    def test_steps_formatted_with_commas(self):
        row = DailyFeatures(date=D1, steps=12345)
        assert "12,345" in build_daily_doc(row).page_content

    def test_workout_omitted_when_none(self):
        row = DailyFeatures(date=D1, workout_count=None)
        assert "workout" not in build_daily_doc(row).page_content

    def test_workout_included_when_present(self):
        row = DailyFeatures(date=D1, workout_count=1, workout_minutes=45.0)
        assert "workout" in build_daily_doc(row).page_content
        assert "45" in build_daily_doc(row).page_content

    def test_sleep_duration_included(self):
        row = DailyFeatures(date=D1, sleep_duration_hrs=7.2)
        assert "7.2h" in build_daily_doc(row).page_content


class TestBuildWeeklyDoc:
    def test_contains_week_range(self):
        row = WeeklySummary(week_start=D1, week_end=D1 + timedelta(days=6))
        content = build_weekly_doc(row).page_content
        assert "2025-01-06" in content
        assert "2025-01-12" in content

    def test_metadata_type(self):
        row = WeeklySummary(week_start=D1, week_end=D1 + timedelta(days=6))
        assert build_weekly_doc(row).metadata["type"] == "weekly"

    def test_avg_steps_included(self):
        row = WeeklySummary(week_start=D1, week_end=D1 + timedelta(days=6), avg_daily_steps=8500.0)
        assert "8,500" in build_weekly_doc(row).page_content

    def test_none_fields_omitted(self):
        row = WeeklySummary(week_start=D1, week_end=D1 + timedelta(days=6))
        content = build_weekly_doc(row).page_content
        assert "steps" not in content
        assert "sleep" not in content


class TestBuildReadinessDoc:
    def test_score_in_content(self):
        row = ReadinessScore(date=D1, score=72.0)
        assert "72" in build_readiness_doc(row).page_content

    def test_metadata_type(self):
        assert build_readiness_doc(ReadinessScore(date=D1, score=72.0)).metadata["type"] == "readiness"

    def test_components_included_when_present(self):
        row = ReadinessScore(date=D1, score=72.0, sleep_score=80.0, hr_score=70.0)
        content = build_readiness_doc(row).page_content
        assert "sleep=80" in content
        assert "HR=70" in content

    def test_explanation_included(self):
        row = ReadinessScore(date=D1, score=72.0, explanation="Good recovery day.")
        assert "Good recovery day." in build_readiness_doc(row).page_content


class TestBuildInsightDoc:
    def test_text_preserved(self):
        row = Insight(generated_for=D1, text="Your RHR is trending down.")
        assert "Your RHR is trending down." in build_insight_doc(row).page_content

    def test_insight_type_in_metadata(self):
        row = Insight(generated_for=D1, insight_type="hr_trend", text="x")
        assert build_insight_doc(row).metadata["insight_type"] == "hr_trend"

    def test_type_in_content(self):
        row = Insight(generated_for=D1, insight_type="hr_trend", text="x")
        assert "hr_trend" in build_insight_doc(row).page_content


class TestBuildRecommendationDoc:
    def test_text_preserved(self):
        row = Recommendation(date=D1, category="recovery", priority=1, text="Rest today.")
        assert "Rest today." in build_recommendation_doc(row).page_content

    def test_category_in_content_and_metadata(self):
        row = Recommendation(date=D1, category="sleep", priority=2, text="x")
        doc = build_recommendation_doc(row)
        assert "sleep" in doc.page_content
        assert doc.metadata["category"] == "sleep"

    def test_reasoning_appended(self):
        row = Recommendation(date=D1, category="recovery", priority=1, text="x", reasoning="readiness < 55")
        assert "readiness < 55" in build_recommendation_doc(row).page_content


# ─── load_all_documents ───────────────────────────────────────────────────────


class TestLoadAllDocuments:
    def test_empty_db_returns_empty(self, db_session):
        assert load_all_documents(db_session) == []

    def test_all_doc_types_present(self, db_session):
        db_session.add(DailyFeatures(date=D1, steps=8000))
        db_session.add(WeeklySummary(week_start=D1, week_end=D1 + timedelta(days=6)))
        db_session.add(ReadinessScore(date=D1, score=72.0))
        db_session.add(Insight(generated_for=D1, text="x"))
        db_session.add(Recommendation(date=D1, category="recovery", text="y"))
        db_session.flush()

        docs = load_all_documents(db_session)
        types = {d.metadata["type"] for d in docs}
        assert types == {"daily", "weekly", "readiness", "insight", "recommendation"}

    def test_count_matches_rows(self, db_session):
        for i in range(3):
            db_session.add(DailyFeatures(date=D1 + timedelta(days=i), steps=8000))
        db_session.flush()
        daily = [d for d in load_all_documents(db_session) if d.metadata["type"] == "daily"]
        assert len(daily) == 3

    def test_documents_are_Document_instances(self, db_session):
        db_session.add(DailyFeatures(date=D1, steps=8000))
        db_session.flush()
        docs = load_all_documents(db_session)
        assert all(isinstance(d, Document) for d in docs)


# ─── Indexer ─────────────────────────────────────────────────────────────────


class TestIndexHealthData:
    def test_empty_db_returns_zero(self, db_session, tmp_path, monkeypatch):
        from backend.rag.indexer import index_health_data
        _redirect_chroma(monkeypatch, tmp_path)
        assert index_health_data(db_session, embeddings=_FakeEmbeddings()) == 0

    def test_returns_correct_count(self, db_session, tmp_path, monkeypatch):
        from backend.rag.indexer import index_health_data
        _redirect_chroma(monkeypatch, tmp_path)
        db_session.add(DailyFeatures(date=D1, steps=8000))
        db_session.add(ReadinessScore(date=D1, score=72.0))
        db_session.flush()
        assert index_health_data(db_session, embeddings=_FakeEmbeddings()) == 2

    def test_idempotent_count(self, db_session, tmp_path, monkeypatch):
        from backend.rag.indexer import index_health_data
        _redirect_chroma(monkeypatch, tmp_path)
        db_session.add(DailyFeatures(date=D1, steps=8000))
        db_session.flush()
        c1 = index_health_data(db_session, embeddings=_FakeEmbeddings())
        c2 = index_health_data(db_session, embeddings=_FakeEmbeddings())
        assert c1 == c2 == 1

    def test_get_vectorstore_returns_chroma(self, tmp_path, monkeypatch):
        from langchain_chroma import Chroma
        from backend.rag.indexer import get_vectorstore
        _redirect_chroma(monkeypatch, tmp_path)
        vs = get_vectorstore(embeddings=_FakeEmbeddings())
        assert isinstance(vs, Chroma)


# ─── Chain ───────────────────────────────────────────────────────────────────


class TestBuildRagChain:
    def test_returns_runnable(self, tmp_path, monkeypatch):
        from backend.rag.chain import build_rag_chain
        _redirect_chroma(monkeypatch, tmp_path)
        chain = build_rag_chain(embeddings=_FakeEmbeddings(), llm=MagicMock())
        assert isinstance(chain, Runnable)

    def test_ask_returns_string(self, tmp_path, monkeypatch):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from backend.rag.chain import ask
        _redirect_chroma(monkeypatch, tmp_path)

        fake_llm = FakeListChatModel(responses=["You slept well."])
        result = ask("How did I sleep?", embeddings=_FakeEmbeddings(), llm=fake_llm)
        assert isinstance(result, str)
        assert len(result) > 0
