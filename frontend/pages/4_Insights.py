import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import empty_state, hero, inject_global_css, pill

st.set_page_config(page_title="Insights | TrainSmartAI", layout="wide")
inject_global_css()

hero(
    "Insights",
    "Correlation-based observations from your history. Regenerated on every upload.",
)

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
TYPE_TONE = {
    "sleep_activity": "blue",
    "rhr_trend": "red",
    "sleep_trend": "violet",
    "workout_pattern": "blue",
    "post_workout_rhr": "red",
    "weekend_sleep": "violet",
    "steps_trend": "green",
}

selected = st.selectbox("Filter by type", INSIGHT_TYPES, index=0)

try:
    rows = api.get_insights(None if selected == "All" else selected)
except requests.RequestException as e:
    st.error(f"Failed to fetch insights: {e}")
    st.stop()

if not rows:
    empty_state(
        "No insights yet",
        "Insights are generated automatically after each upload. Try uploading more data or changing the filter.",
    )
    st.stop()

rows.sort(key=lambda r: r.get("generated_for", ""), reverse=True)

for r in rows:
    label = r.get("insight_type") or "general"
    tone = TYPE_TONE.get(label, "slate")
    with st.container(border=True):
        head_l, head_r = st.columns([1, 5])
        with head_l:
            st.markdown(pill(label.replace("_", " "), tone), unsafe_allow_html=True)
            st.caption(r.get("generated_for", ""))
        with head_r:
            st.write(r["text"])
