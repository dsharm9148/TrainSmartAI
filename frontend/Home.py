import streamlit as st

st.set_page_config(
    page_title="TrainSmartAI",
    page_icon="🏃",
    layout="wide",
)

st.title("TrainSmartAI")
st.subheader("Your personal health intelligence dashboard")

st.markdown(
    """
    Built on Apple Health export data. Inspired by Whoop.

    **Get started:** Upload your Apple Health export on the Upload page to begin.

    ---

    **Pages**
    - **Upload** — ingest your Apple Health export XML
    - **Dashboard** — daily overview of steps, sleep, heart rate, and workouts
    - **Trends** — rolling charts and week-over-week comparisons
    - **Insights** — data-driven observations from your history
    - **Recommendations** — personalized suggestions based on your patterns
    - **Readiness** — your daily readiness score and what drives it
    - **Assistant** — chat with an LLM coach grounded in your own data
    - **Day Types** — cluster analysis of your day patterns
    """
)
