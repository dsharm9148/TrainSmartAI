import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Dashboard | TrainSmartAI", layout="wide")
st.title("Dashboard")
st.caption("Daily and weekly trends from your ingested health data.")

# ─── Date range selector ────────────────────────────────────────────────────

today = date.today()
default_start = today - timedelta(days=89)

c1, c2 = st.columns(2)
start = c1.date_input("Start date", value=default_start)
end = c2.date_input("End date", value=today)

if start > end:
    st.error("Start date must be on or before end date.")
    st.stop()

# ─── Fetch ──────────────────────────────────────────────────────────────────

try:
    daily = api.get_daily(start.isoformat(), end.isoformat())
    weekly = api.get_weekly(start.isoformat(), end.isoformat())
except requests.RequestException as e:
    st.error(f"Failed to fetch data from API: {e}")
    st.stop()

if not daily:
    st.info("No daily features in this range. Upload an export from the Upload page.")
    st.stop()

df = pd.DataFrame(daily)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── KPI strip ──────────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
k1.metric("Days in range", len(df))
if df["steps"].notna().any():
    k2.metric("Avg steps/day", f"{df['steps'].dropna().mean():,.0f}")
if df["sleep_duration_hrs"].notna().any():
    k3.metric("Avg sleep (hrs)", f"{df['sleep_duration_hrs'].dropna().mean():.1f}")
if df["resting_heart_rate"].notna().any():
    k4.metric("Avg resting HR", f"{df['resting_heart_rate'].dropna().mean():.0f} bpm")

st.divider()

# ─── Charts ─────────────────────────────────────────────────────────────────

tab_daily, tab_weekly = st.tabs(["Daily trends", "Weekly summaries"])

with tab_daily:
    if df["steps"].notna().any():
        fig = px.line(df, x="date", y="steps", title="Daily steps", markers=True)
        fig.add_hline(y=8000, line_dash="dot", annotation_text="8k target")
        st.plotly_chart(fig, use_container_width=True)

    if df["sleep_duration_hrs"].notna().any():
        fig = px.line(
            df,
            x="date",
            y="sleep_duration_hrs",
            title="Sleep duration (hrs)",
            markers=True,
        )
        fig.add_hline(y=7.0, line_dash="dot", annotation_text="7h target")
        st.plotly_chart(fig, use_container_width=True)

    if df["resting_heart_rate"].notna().any():
        fig = px.line(
            df,
            x="date",
            y="resting_heart_rate",
            title="Resting heart rate (bpm)",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    if df["workout_minutes"].notna().any():
        fig = px.bar(df, x="date", y="workout_minutes", title="Workout minutes per day")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw daily features"):
        st.dataframe(df, use_container_width=True)

with tab_weekly:
    if not weekly:
        st.info("No weekly summaries in this range yet.")
    else:
        wdf = pd.DataFrame(weekly)
        wdf["week_start"] = pd.to_datetime(wdf["week_start"])
        wdf = wdf.sort_values("week_start")

        if wdf["avg_daily_steps"].notna().any():
            fig = px.bar(wdf, x="week_start", y="avg_daily_steps", title="Avg daily steps by week")
            st.plotly_chart(fig, use_container_width=True)

        if wdf["avg_sleep_hrs"].notna().any():
            fig = px.line(
                wdf,
                x="week_start",
                y="avg_sleep_hrs",
                title="Avg sleep hours per week",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        if wdf["sleep_consistency_score"].notna().any():
            fig = px.bar(
                wdf,
                x="week_start",
                y="sleep_consistency_score",
                title="Sleep consistency score (0-100)",
            )
            st.plotly_chart(fig, use_container_width=True)

        if wdf["avg_readiness_score"].notna().any():
            fig = px.line(
                wdf,
                x="week_start",
                y="avg_readiness_score",
                title="Avg readiness by week",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw weekly summaries"):
            st.dataframe(wdf, use_container_width=True)
