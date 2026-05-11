import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Readiness | TrainSmartAI", layout="wide")
st.title("Readiness")
st.caption(
    "Daily readiness score (0-100). Weighted across sleep (40%), HR vs your "
    "baseline (30%), training load (20%), and bedtime consistency (10%)."
)

today = date.today()
default_start = today - timedelta(days=29)
c1, c2 = st.columns(2)
start = c1.date_input("Start date", value=default_start)
end = c2.date_input("End date", value=today)

if start > end:
    st.error("Start date must be on or before end date.")
    st.stop()

try:
    rows = api.get_readiness(start.isoformat(), end.isoformat())
except requests.RequestException as e:
    st.error(f"Failed to fetch readiness data: {e}")
    st.stop()

if not rows:
    st.info(
        "No readiness scores in this range. Upload data from the Upload page first."
    )
    st.stop()

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── Today / latest snapshot ────────────────────────────────────────────────

latest = df.iloc[-1]

st.subheader(f"Latest score — {latest['date'].date()}")

main_col, gauge_col = st.columns([2, 1])
with main_col:
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Sleep", f"{latest['sleep_score']:.0f}" if pd.notna(latest['sleep_score']) else "—")
    cc2.metric("HR", f"{latest['hr_score']:.0f}" if pd.notna(latest['hr_score']) else "—")
    cc3.metric("Load", f"{latest['load_score']:.0f}" if pd.notna(latest['load_score']) else "—")
    cc4.metric(
        "Consistency",
        f"{latest['consistency_score']:.0f}" if pd.notna(latest['consistency_score']) else "—",
    )
    if latest.get("explanation"):
        st.info(latest["explanation"])

with gauge_col:
    score = float(latest["score"])
    bar_color = (
        "#16a34a" if score >= 75 else "#eab308" if score >= 55 else "#dc2626"
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Today's score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 55], "color": "#fee2e2"},
                    {"range": [55, 75], "color": "#fef9c3"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(gauge, use_container_width=True)

st.divider()

# ─── Trend chart ────────────────────────────────────────────────────────────

fig = px.line(df, x="date", y="score", title="Readiness over time", markers=True)
fig.add_hline(y=75, line_dash="dot", line_color="#16a34a", annotation_text="Ready (75)")
fig.add_hline(y=55, line_dash="dot", line_color="#dc2626", annotation_text="Caution (55)")
st.plotly_chart(fig, use_container_width=True)

# ─── Components stacked ─────────────────────────────────────────────────────

with st.expander("Component breakdown over time"):
    component_cols = ["sleep_score", "hr_score", "load_score", "consistency_score"]
    long = df[["date"] + component_cols].melt("date", var_name="component", value_name="value")
    fig = px.line(
        long.dropna(),
        x="date",
        y="value",
        color="component",
        title="Component scores",
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw readiness data"):
    st.dataframe(df, use_container_width=True)
