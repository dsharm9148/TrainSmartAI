"""
Thin HTTP client for the TrainSmartAI FastAPI backend.

Streamlit pages import these helpers rather than hand-building requests so
the API base URL and timeout handling stay in one place.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

BASE_URL = os.environ.get("TRAINSMART_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 30
CHAT_TIMEOUT = 120  # LLM responses can take a while on first run


def _url(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def health() -> dict:
    r = requests.get(_url("/health"), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def upload_xml(filename: str, contents: bytes) -> dict:
    r = requests.post(
        _url("/upload"),
        files={"file": (filename, contents, "application/xml")},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def get_daily(start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[dict]:
    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    r = requests.get(_url("/daily"), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_weekly(start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[dict]:
    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    r = requests.get(_url("/weekly"), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_readiness(start_date: Optional[str] = None, end_date: Optional[str] = None) -> list[dict]:
    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    r = requests.get(_url("/readiness"), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_insights(insight_type: Optional[str] = None) -> list[dict]:
    params: dict[str, Any] = {}
    if insight_type:
        params["insight_type"] = insight_type
    r = requests.get(_url("/insights"), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_recommendations(
    for_date: Optional[str] = None, category: Optional[str] = None
) -> list[dict]:
    params: dict[str, Any] = {}
    if for_date:
        params["for_date"] = for_date
    if category:
        params["category"] = category
    r = requests.get(_url("/recommendations"), params=params, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_chat(message: str, session_id: Optional[str] = None) -> dict:
    body: dict[str, Any] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    r = requests.post(_url("/chat"), json=body, timeout=CHAT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_clusters() -> list[dict]:
    r = requests.get(_url("/clusters"), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_clusters_recompute(n_clusters: int = 4) -> dict:
    r = requests.post(
        _url("/clusters/recompute"),
        params={"n_clusters": n_clusters},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def list_chat_sessions() -> list[dict]:
    r = requests.get(_url("/chat/sessions"), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_chat_messages(session_id: str) -> list[dict]:
    r = requests.get(_url(f"/chat/sessions/{session_id}/messages"), timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()
