import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import (
    date_range_picker,
    empty_state,
    hero,
    inject_global_css,
    pill,
    readiness_tone,
)

st.set_page_config(page_title="Readiness | TrainSmartAI", layout="wide")
inject_global_css()

hero(
    "Readiness",
    "Weighted score across sleep (40%), HR vs personal baseline (30%), training load (20%), and bedtime consistency (10%).",
)

start, end = date_range_picker(default_lookback_days=30, key_prefix="ready")

try:
    rows = api.get_readiness(start.isoformat(), end.isoformat())
except requests.RequestException as e:
    st.error(f"Failed to fetch readiness data: {e}")
    st.stop()

if not rows:
    empty_state("No readiness scores yet", "Upload data from the Upload page first.")
    st.stop()

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── Latest snapshot ────────────────────────────────────────────────────────

latest = df.iloc[-1]
score = float(latest["score"])
tone = readiness_tone(score)
bar_color = {"green": "#16a34a", "amber": "#eab308", "red": "#dc2626"}.get(tone, "#64748b")

snap_l, snap_r = st.columns([1, 2])

with snap_l:
    with st.container(border=True):
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"Today — {latest['date'].date()}", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": bar_color, "thickness": 0.25},
                    "steps": [
                        {"range": [0, 55], "color": "#fee2e2"},
                        {"range": [55, 75], "color": "#fef9c3"},
                        {"range": [75, 100], "color": "#dcfce7"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 2},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )
        gauge.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(gauge, use_container_width=True)

with snap_r:
    with st.container(border=True):
        st.markdown("**Component breakdown**")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Sleep", f"{latest['sleep_score']:.0f}" if pd.notna(latest['sleep_score']) else "—")
        cc2.metric("HR", f"{latest['hr_score']:.0f}" if pd.notna(latest['hr_score']) else "—")
        cc3.metric("Load", f"{latest['load_score']:.0f}" if pd.notna(latest['load_score']) else "—")
        cc4.metric(
            "Consistency",
            f"{latest['consistency_score']:.0f}" if pd.notna(latest['consistency_score']) else "—",
        )
        if latest.get("explanation"):
            st.markdown(pill("Why", tone), unsafe_allow_html=True)
            st.caption(latest["explanation"])

# ─── Trend ──────────────────────────────────────────────────────────────────

st.subheader("Trend")
fig = px.line(df, x="date", y="score", markers=True, template="simple_white")
fig.update_traces(line_color="#0f172a")
fig.add_hrect(y0=75, y1=100, fillcolor="#16a34a", opacity=0.08, line_width=0)
fig.add_hrect(y0=55, y1=75, fillcolor="#eab308", opacity=0.08, line_width=0)
fig.add_hrect(y0=0, y1=55, fillcolor="#dc2626", opacity=0.08, line_width=0)
fig.add_hline(y=75, line_dash="dot", line_color="#16a34a")
fig.add_hline(y=55, line_dash="dot", line_color="#dc2626")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

# ─── Components over time ───────────────────────────────────────────────────

with st.expander("Component scores over time"):
    component_cols = ["sleep_score", "hr_score", "load_score", "consistency_score"]
    long = df[["date"] + component_cols].melt("date", var_name="component", value_name="value")
    fig = px.line(long.dropna(), x="date", y="value", color="component", template="simple_white")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw readiness data"):
    st.dataframe(df, use_container_width=True)
