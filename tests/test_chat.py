"""
Tests for the chat API — POST /chat, GET /chat/sessions,
GET /chat/sessions/{id}/messages.

The RAG chain is replaced with a stub via the _chain_runner indirection
so no OpenAI calls happen during tests.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.routes import chat as chat_route
from backend.db.models import ChatMessage, ChatSession
from backend.db.session import get_db


def _db_override(session):
    def _dep():
        yield session
    return _dep


@pytest.fixture
def stub_chain(monkeypatch):
    """Replace the RAG chain with a deterministic stub."""
    calls: list[dict] = []

    def fake_ask(message: str, history=None, **kw) -> str:
        calls.append({"message": message, "history": history or []})
        return f"echo: {message}"

    monkeypatch.setattr(chat_route, "_chain_runner", lambda: fake_ask)
    return calls


@pytest.fixture
def client(db_session, stub_chain):
    app.dependency_overrides[get_db] = _db_override(db_session)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ─── POST /chat ─────────────────────────────────────────────────────────────


class TestPostChat:
    def test_creates_session_when_id_absent(self, client, db_session):
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert "session_id" in body
        assert body["response"] == "echo: hi"
        assert db_session.query(ChatSession).count() == 1

    def test_persists_user_and_assistant_messages(self, client, db_session):
        client.post("/chat", json={"message": "how am I"})
        rows = db_session.query(ChatMessage).order_by(ChatMessage.created_at).all()
        assert [r.role for r in rows] == ["user", "assistant"]
        assert rows[0].content == "how am I"
        assert rows[1].content == "echo: how am I"

    def test_continuing_session_reuses_id(self, client, db_session):
        first = client.post("/chat", json={"message": "hi"}).json()
        sid = first["session_id"]
        second = client.post("/chat", json={"message": "again", "session_id": sid}).json()
        assert second["session_id"] == sid
        assert db_session.query(ChatSession).count() == 1
        assert db_session.query(ChatMessage).count() == 4

    def test_history_passed_to_chain_on_second_turn(self, client, stub_chain):
        first = client.post("/chat", json={"message": "turn 1"}).json()
        client.post("/chat", json={"message": "turn 2", "session_id": first["session_id"]})
        # second call to fake_ask gets history with 2 messages: user "turn 1", assistant "echo: turn 1"
        last_call = stub_chain[-1]
        assert len(last_call["history"]) == 2
        assert last_call["history"][0].content == "turn 1"
        assert last_call["history"][1].content == "echo: turn 1"

    def test_rejects_empty_message(self, client):
        assert client.post("/chat", json={"message": ""}).status_code == 400
        assert client.post("/chat", json={"message": "   "}).status_code == 400

    def test_unknown_session_id_returns_404(self, client):
        resp = client.post(
            "/chat",
            json={"message": "hi", "session_id": str(uuid4())},
        )
        assert resp.status_code == 404


# ─── GET /chat/sessions ─────────────────────────────────────────────────────


class TestListSessions:
    def test_empty_returns_empty_list(self, client):
        assert client.get("/chat/sessions").json() == []

    def test_returns_created_session(self, client):
        client.post("/chat", json={"message": "hi"})
        rows = client.get("/chat/sessions").json()
        assert len(rows) == 1
        assert "session_id" in rows[0]
        assert "started_at" in rows[0]

    def test_newest_first(self, client, db_session):
        # Two sessions
        r1 = client.post("/chat", json={"message": "a"}).json()
        r2 = client.post("/chat", json={"message": "b"}).json()
        rows = client.get("/chat/sessions").json()
        # r2 created after r1 → r2 first
        assert rows[0]["session_id"] == r2["session_id"]
        assert rows[1]["session_id"] == r1["session_id"]


# ─── GET /chat/sessions/{id}/messages ───────────────────────────────────────


class TestListMessages:
    def test_returns_messages_oldest_first(self, client):
        sid = client.post("/chat", json={"message": "first"}).json()["session_id"]
        client.post("/chat", json={"message": "second", "session_id": sid})

        rows = client.get(f"/chat/sessions/{sid}/messages").json()
        assert [r["role"] for r in rows] == ["user", "assistant", "user", "assistant"]
        assert rows[0]["content"] == "first"
        assert rows[2]["content"] == "second"

    def test_unknown_session_returns_404(self, client):
        resp = client.get(f"/chat/sessions/{uuid4()}/messages")
        assert resp.status_code == 404


# ─── History limit ───────────────────────────────────────────────────────────


class TestHistoryLimit:
    def test_history_capped_at_limit(self, client, stub_chain, monkeypatch):
        monkeypatch.setattr(chat_route, "HISTORY_LIMIT", 4)
        sid = None
        for i in range(6):
            resp = client.post(
                "/chat",
                json={"message": f"msg {i}", "session_id": sid},
            ).json()
            sid = resp["session_id"]
        # 6 turns → 12 stored messages, but history passed on turn 6
        # should be capped at HISTORY_LIMIT (= 4)
        last_history = stub_chain[-1]["history"]
        assert len(last_history) <= 4
