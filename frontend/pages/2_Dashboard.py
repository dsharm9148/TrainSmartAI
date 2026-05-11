import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import date_range_picker, empty_state, hero, inject_global_css

st.set_page_config(page_title="Dashboard | TrainSmartAI", layout="wide")
inject_global_css()

hero("Dashboard", "Daily and weekly trends from your ingested health data.")

start, end = date_range_picker(default_lookback_days=90, key_prefix="dash")

# ─── Fetch ──────────────────────────────────────────────────────────────────

try:
    daily = api.get_daily(start.isoformat(), end.isoformat())
    weekly = api.get_weekly(start.isoformat(), end.isoformat())
except requests.RequestException as e:
    st.error(f"Failed to fetch data from API: {e}")
    st.stop()

if not daily:
    empty_state(
        "No data in this range",
        "Upload an export from the Upload page or pick a wider date range.",
    )
    st.stop()

df = pd.DataFrame(daily)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── KPI strip ──────────────────────────────────────────────────────────────

with st.container(border=True):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Days", len(df))
    k2.metric(
        "Avg steps",
        f"{df['steps'].dropna().mean():,.0f}" if df["steps"].notna().any() else "—",
    )
    k3.metric(
        "Avg sleep",
        f"{df['sleep_duration_hrs'].dropna().mean():.1f} h"
        if df["sleep_duration_hrs"].notna().any() else "—",
    )
    k4.metric(
        "Avg RHR",
        f"{df['resting_heart_rate'].dropna().mean():.0f} bpm"
        if df["resting_heart_rate"].notna().any() else "—",
    )
    workout_total = df["workout_minutes"].fillna(0).sum()
    k5.metric("Workout min total", f"{workout_total:,.0f}")

# ─── Charts ─────────────────────────────────────────────────────────────────

tab_daily, tab_weekly = st.tabs(["Daily trends", "Weekly summaries"])

PLOT_TEMPLATE = "simple_white"
ACCENT = "#16a34a"


def _chart(fig):
    fig.update_layout(template=PLOT_TEMPLATE, margin=dict(l=10, r=10, t=50, b=10), height=350)
    st.plotly_chart(fig, use_container_width=True)


with tab_daily:
    grid_a, grid_b = st.columns(2)

    with grid_a:
        if df["steps"].notna().any():
            fig = px.line(df, x="date", y="steps", title="Daily steps", markers=True)
            fig.update_traces(line_color=ACCENT)
            fig.add_hline(y=8000, line_dash="dot", line_color="#94a3b8", annotation_text="8k target")
            _chart(fig)

    with grid_b:
        if df["sleep_duration_hrs"].notna().any():
            fig = px.line(df, x="date", y="sleep_duration_hrs", title="Sleep duration (hrs)", markers=True)
            fig.update_traces(line_color="#7c3aed")
            fig.add_hline(y=7.0, line_dash="dot", line_color="#94a3b8", annotation_text="7h target")
            _chart(fig)

    grid_c, grid_d = st.columns(2)
    with grid_c:
        if df["resting_heart_rate"].notna().any():
            fig = px.line(df, x="date", y="resting_heart_rate", title="Resting heart rate (bpm)", markers=True)
            fig.update_traces(line_color="#dc2626")
            _chart(fig)
    with grid_d:
        if df["workout_minutes"].notna().any():
            fig = px.bar(df, x="date", y="workout_minutes", title="Workout minutes per day")
            fig.update_traces(marker_color="#2563eb")
            _chart(fig)

    with st.expander("Raw daily features"):
        st.dataframe(df, use_container_width=True)

with tab_weekly:
    if not weekly:
        empty_state("No weekly summaries", "Weekly summaries appear after at least one full ISO week of data.")
    else:
        wdf = pd.DataFrame(weekly)
        wdf["week_start"] = pd.to_datetime(wdf["week_start"])
        wdf = wdf.sort_values("week_start")

        g1, g2 = st.columns(2)
        with g1:
            if wdf["avg_daily_steps"].notna().any():
                fig = px.bar(wdf, x="week_start", y="avg_daily_steps", title="Avg daily steps by week")
                fig.update_traces(marker_color=ACCENT)
                _chart(fig)
        with g2:
            if wdf["avg_sleep_hrs"].notna().any():
                fig = px.line(wdf, x="week_start", y="avg_sleep_hrs", title="Avg sleep hours per week", markers=True)
                fig.update_traces(line_color="#7c3aed")
                _chart(fig)

        g3, g4 = st.columns(2)
        with g3:
            if wdf["sleep_consistency_score"].notna().any():
                fig = px.bar(wdf, x="week_start", y="sleep_consistency_score", title="Sleep consistency score (0-100)")
                fig.update_traces(marker_color="#0ea5e9")
                _chart(fig)
        with g4:
            if wdf["avg_readiness_score"].notna().any():
                fig = px.line(wdf, x="week_start", y="avg_readiness_score", title="Avg readiness by week", markers=True)
                fig.update_traces(line_color="#16a34a")
                _chart(fig)

        with st.expander("Raw weekly summaries"):
            st.dataframe(wdf, use_container_width=True)
