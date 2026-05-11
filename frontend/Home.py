import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(
    page_title="TrainSmartAI",
    page_icon="🏃",
    layout="wide",
)

st.title("TrainSmartAI")
st.caption("Your personal health intelligence dashboard — built on Apple Health export data.")

# ─── Backend status ──────────────────────────────────────────────────────────

col1, col2 = st.columns([1, 3])
with col1:
    try:
        h = api.health()
        if h.get("database") == "connected":
            st.success("Backend: OK")
        else:
            st.warning(f"Backend: DB {h.get('database')}")
    except requests.RequestException as e:
        st.error(f"Backend unreachable at {api.BASE_URL}")
        st.caption(str(e))

with col2:
    st.markdown(f"`API base URL`: **{api.BASE_URL}**")
    st.caption("Override via the `TRAINSMART_API_URL` environment variable.")

st.divider()

# ─── Quick stats ─────────────────────────────────────────────────────────────

try:
    daily = api.get_daily()
    weekly = api.get_weekly()
    readiness = api.get_readiness()
    recs = api.get_recommendations()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days tracked", len(daily))
    c2.metric("Weeks summarised", len(weekly))
    c3.metric("Readiness scores", len(readiness))
    c4.metric("Recommendations", len(recs))

    if daily:
        last = daily[-1]
        st.caption(f"Most recent day in DB: **{last['date']}**")
except requests.RequestException:
    st.info("Upload an Apple Health export from the Upload page to get started.")

st.divider()

# ─── Page directory ──────────────────────────────────────────────────────────

st.markdown(
    """
    **Pages**

    - **Upload** — ingest your Apple Health export XML
    - **Dashboard** — daily and weekly trends with charts
    - **Readiness** — daily readiness score with component breakdowns
    - **Insights** — correlation-based observations from your history
    - **Recommendations** — rule-based suggestions by category and priority
    - **Chat** — ask freeform questions; the local LLM grounds answers in your data
    """
)
