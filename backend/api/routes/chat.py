"""
Chat API — multi-turn assistant with persistent conversation history.

POST /chat creates a session on first call, then accepts the returned
session_id on subsequent turns to keep context. Every user/assistant
exchange is persisted to chat_sessions / chat_messages.
"""
from __future__ import annotations

from typing import Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.api.schemas import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionOut,
)
from backend.db.models import ChatMessage, ChatSession
from backend.db.session import get_db
from backend.rag.chain import ask, history_from_rows

router = APIRouter()

# How many prior turns to feed back into the chain. Cap keeps prompt size sane.
HISTORY_LIMIT = 10


def _chain_runner() -> Callable[..., str]:
    """Indirection point so tests can monkey-patch the LLM call."""
    return ask


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Send a message; return assistant reply and the session id."""
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session = _resolve_session(db, req.session_id)

    prior_rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.session_id)
        .order_by(ChatMessage.id.desc())
        .limit(HISTORY_LIMIT)
        .all()
    )
    history = history_from_rows(list(reversed(prior_rows)))

    runner = _chain_runner()
    answer = runner(req.message, history=history)

    db.add(ChatMessage(session_id=session.session_id, role="user", content=req.message))
    db.add(ChatMessage(session_id=session.session_id, role="assistant", content=answer))
    db.commit()

    return ChatResponse(session_id=session.session_id, response=answer)


@router.get("/chat/sessions", response_model=list[ChatSessionOut], tags=["chat"])
def list_sessions(db: Session = Depends(get_db)) -> list[ChatSessionOut]:
    """Return every chat session, newest first."""
    return (
        db.query(ChatSession)
        .order_by(ChatSession.last_active.desc(), ChatSession.id.desc())
        .all()
    )


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=list[ChatMessageOut],
    tags=["chat"],
)
def list_messages(session_id: UUID, db: Session = Depends(get_db)) -> list[ChatMessageOut]:
    """Return all messages for a session, oldest first."""
    session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
        .all()
    )


def _resolve_session(db: Session, session_id: Optional[UUID]) -> ChatSession:
    if session_id is not None:
        existing = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="session not found")
        return existing

    session = ChatSession()
    db.add(session)
    db.flush()
    return session
