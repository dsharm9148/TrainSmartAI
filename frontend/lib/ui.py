"""
Shared UI helpers — keeps every Streamlit page visually consistent.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import streamlit as st

# ─── Global CSS ─────────────────────────────────────────────────────────────

_CSS = """
<style>
  /* Tighter top padding */
  .block-container { padding-top: 2rem; padding-bottom: 4rem; }

  /* Larger, denser metric labels */
  div[data-testid="stMetricLabel"] p {
    font-size: 0.85rem; font-weight: 500; color: #475569;
  }
  div[data-testid="stMetricValue"] {
    font-size: 1.75rem; font-weight: 600;
  }

  /* Page hero block */
  .ts-hero {
    padding: 1.25rem 1.5rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .ts-hero h1 { color: #f1f5f9; margin: 0 0 .25rem 0; font-size: 1.6rem; }
  .ts-hero p  { color: #cbd5e1; margin: 0; font-size: 0.95rem; }

  /* Coloured status pills */
  .ts-pill {
    display: inline-block; padding: .2rem .7rem; border-radius: 999px;
    font-size: .8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .ts-pill.green  { background: #dcfce7; color: #166534; }
  .ts-pill.amber  { background: #fef3c7; color: #92400e; }
  .ts-pill.red    { background: #fee2e2; color: #991b1b; }
  .ts-pill.blue   { background: #dbeafe; color: #1e40af; }
  .ts-pill.violet { background: #ede9fe; color: #5b21b6; }
  .ts-pill.slate  { background: #e2e8f0; color: #334155; }

  /* Card-like containers (used with st.container(border=True)) */
  [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px !important;
  }

  /* Sidebar polish */
  section[data-testid="stSidebar"] {
    background: #f8fafc;
  }
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: Optional[str] = None) -> None:
    """Page header banner — dark gradient with title + subtitle."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="ts-hero"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "slate") -> str:
    """Return HTML for a coloured status pill (use with unsafe_allow_html)."""
    tone = tone if tone in {"green", "amber", "red", "blue", "violet", "slate"} else "slate"
    return f'<span class="ts-pill {tone}">{label}</span>'


def readiness_tone(score: Optional[float]) -> str:
    if score is None:
        return "slate"
    if score >= 75:
        return "green"
    if score >= 55:
        return "amber"
    return "red"


def priority_tone(priority: Optional[int]) -> str:
    return {1: "red", 2: "amber", 3: "blue"}.get(priority or 0, "slate")


def category_tone(category: Optional[str]) -> str:
    return {
        "recovery": "red",
        "workout": "blue",
        "sleep": "violet",
        "habit": "green",
    }.get((category or "").lower(), "slate")


def date_range_picker(
    default_lookback_days: int = 30,
    key_prefix: str = "range",
) -> tuple[date, date]:
    today = date.today()
    default_start = today - timedelta(days=default_lookback_days - 1)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", value=default_start, key=f"{key_prefix}_start")
    end = c2.date_input("End date", value=today, key=f"{key_prefix}_end")
    if start > end:
        st.error("Start date must be on or before end date.")
        st.stop()
    return start, end


def empty_state(title: str, body: str, action_hint: Optional[str] = None) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.write(body)
        if action_hint:
            st.caption(action_hint)
