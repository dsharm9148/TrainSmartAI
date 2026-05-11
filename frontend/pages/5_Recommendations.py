import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date

import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Recommendations | TrainSmartAI", layout="wide")
st.title("Recommendations")
st.caption(
    "Rule-based suggestions tailored to your recent data. Each rule explains "
    "why it fired."
)

CATEGORIES = ["All", "recovery", "workout", "sleep", "habit"]
CATEGORY_COLOR = {
    "recovery": "#dc2626",
    "workout": "#2563eb",
    "sleep": "#7c3aed",
    "habit": "#16a34a",
}

c1, c2 = st.columns(2)
with c1:
    category = st.selectbox("Category", CATEGORIES, index=0)
with c2:
    pick_date = st.date_input("On date", value=date.today())
    use_date = st.checkbox("Filter by this date", value=False)

try:
    rows = api.get_recommendations(
        for_date=pick_date.isoformat() if use_date else None,
        category=None if category == "All" else category,
    )
except requests.RequestException as e:
    st.error(f"Failed to fetch recommendations: {e}")
    st.stop()

if not rows:
    st.info("No recommendations match this filter.")
    st.stop()

# ─── Group by priority ──────────────────────────────────────────────────────

by_pri: dict[int, list[dict]] = {1: [], 2: [], 3: []}
for r in rows:
    by_pri.setdefault(r.get("priority") or 2, []).append(r)

PRIORITY_LABEL = {1: "Priority 1 — act today", 2: "Priority 2 — worth doing", 3: "Priority 3 — nudge"}

for pri in (1, 2, 3):
    if not by_pri.get(pri):
        continue
    st.subheader(PRIORITY_LABEL[pri])
    for r in by_pri[pri]:
        cat = r.get("category") or "general"
        color = CATEGORY_COLOR.get(cat, "#64748b")
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(
                    f"<span style='color:{color}; font-weight:600'>{cat.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(r.get("date", ""))
            with c2:
                st.write(r["text"])
                if r.get("reasoning"):
                    st.caption(f"Why: {r['reasoning']}")
