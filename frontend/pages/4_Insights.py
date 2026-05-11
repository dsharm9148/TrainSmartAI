import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Insights | TrainSmartAI", layout="wide")
st.title("Insights")
st.caption(
    "Correlation-based observations from your history. Insights regenerate "
    "on every upload — older runs are replaced for the same date."
)

# ─── Filter ─────────────────────────────────────────────────────────────────

INSIGHT_TYPES = [
    "All",
    "sleep_activity",
    "rhr_trend",
    "sleep_trend",
    "workout_pattern",
    "post_workout_rhr",
    "weekend_sleep",
    "steps_trend",
]

selected = st.selectbox("Filter by type", INSIGHT_TYPES, index=0)

try:
    rows = api.get_insights(None if selected == "All" else selected)
except requests.RequestException as e:
    st.error(f"Failed to fetch insights: {e}")
    st.stop()

if not rows:
    st.info(
        "No insights match this filter. Insights are generated automatically "
        "after each upload — try uploading more data, or pick another filter."
    )
    st.stop()

# Newest first
rows.sort(key=lambda r: r.get("generated_for", ""), reverse=True)

# ─── Render as cards ────────────────────────────────────────────────────────

for r in rows:
    label = r.get("insight_type") or "general"
    with st.container(border=True):
        c1, c2 = st.columns([1, 4])
        c1.markdown(f"**{label}**")
        c1.caption(r.get("generated_for", ""))
        c2.write(r["text"])
