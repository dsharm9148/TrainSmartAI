import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import (
    hero,
    inject_global_css,
    pill,
    readiness_tone,
)

st.set_page_config(
    page_title="TrainSmartAI",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

hero(
    "TrainSmartAI",
    "Personal health intelligence built on Apple Health export data — readiness, insights, recommendations, and a local-LLM chat coach.",
)

# ─── Backend status row ─────────────────────────────────────────────────────

status_col, url_col = st.columns([1, 3])
with status_col:
    try:
        h = api.health()
        if h.get("database") == "connected":
            st.markdown(pill("Backend OK", "green"), unsafe_allow_html=True)
        else:
            st.markdown(pill(f"DB {h.get('database')}", "amber"), unsafe_allow_html=True)
    except requests.RequestException:
        st.markdown(pill("Backend unreachable", "red"), unsafe_allow_html=True)
        st.caption(f"Expected at `{api.BASE_URL}`")

with url_col:
    st.caption(
        f"API: `{api.BASE_URL}` · "
        "override via the `TRAINSMART_API_URL` environment variable"
    )

st.divider()

# ─── At-a-glance counts ─────────────────────────────────────────────────────

try:
    daily = api.get_daily()
    weekly = api.get_weekly()
    readiness = api.get_readiness()
    recs = api.get_recommendations()
    insights = api.get_insights()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Days tracked", len(daily))
    c2.metric("Weeks summarised", len(weekly))
    c3.metric("Readiness scores", len(readiness))
    c4.metric("Insights", len(insights))
    c5.metric("Recommendations", len(recs))
except requests.RequestException:
    daily = readiness = recs = []
    st.warning("API not reachable — start it with `uvicorn backend.api.main:app --reload`.")

# ─── Latest snapshot ────────────────────────────────────────────────────────

if readiness:
    latest = sorted(readiness, key=lambda r: r["date"])[-1]
    score = latest["score"]
    tone = readiness_tone(score)

    st.subheader("Latest readiness")
    snap_a, snap_b = st.columns([1, 3])
    with snap_a:
        with st.container(border=True):
            st.metric("Today's score", f"{score:.0f} / 100")
            st.markdown(pill(latest["date"], tone), unsafe_allow_html=True)
    with snap_b:
        with st.container(border=True):
            st.markdown("**Component breakdown**")
            cols = st.columns(4)
            cols[0].metric("Sleep", f"{latest.get('sleep_score') or 0:.0f}")
            cols[1].metric("HR", f"{latest.get('hr_score') or 0:.0f}")
            cols[2].metric("Load", f"{latest.get('load_score') or 0:.0f}")
            cols[3].metric("Consistency", f"{latest.get('consistency_score') or 0:.0f}")
            if latest.get("explanation"):
                st.caption(latest["explanation"])

if recs:
    st.subheader("Top recommendations")
    # Sort by date desc, priority asc, take top 3
    top = sorted(recs, key=lambda r: (r["date"], r.get("priority") or 9), reverse=False)
    top = sorted(top, key=lambda r: (r["date"], -(r.get("priority") or 9)), reverse=True)[:3]
    for r in top:
        with st.container(border=True):
            head_a, head_b = st.columns([2, 8])
            with head_a:
                st.markdown(
                    pill(r.get("category", "n/a"), {
                        "recovery": "red", "workout": "blue",
                        "sleep": "violet", "habit": "green",
                    }.get(r.get("category"), "slate")),
                    unsafe_allow_html=True,
                )
                st.caption(r["date"])
            with head_b:
                st.write(r["text"])

if not daily:
    st.info(
        "No data yet — head to the **Upload** page in the sidebar to ingest "
        "your Apple Health export, or generate synthetic demo data with "
        "`python data/synthetic/generate_sample.py`."
    )

st.divider()

# ─── Page directory ─────────────────────────────────────────────────────────

st.markdown(
    """
    ### Pages

    | Page | What you'll see |
    |---|---|
    | **Upload** | Drop your Apple Health export XML; the full pipeline runs end-to-end |
    | **Dashboard** | Daily and weekly trends — steps, sleep, HR, workout load |
    | **Readiness** | Daily score with sleep / HR / load / consistency breakdowns |
    | **Insights** | Correlation-based observations from your history |
    | **Recommendations** | Rule-based suggestions grouped by priority and category |
    | **Chat** | Ask freeform questions — the local Ollama LLM grounds answers in your data |
    | **Day Types** | K-means archetype labels for every day |
    """
)
