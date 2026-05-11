import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Day Types | TrainSmartAI", layout="wide")
st.title("Day Types")
st.caption(
    "K-means clusters your days by activity, sleep, resting HR, and workout "
    "load. Each centroid gets an automatic archetype label."
)

# ─── Fetch ──────────────────────────────────────────────────────────────────

try:
    rows = api.get_clusters()
except requests.RequestException as e:
    st.error(f"Failed to fetch clusters: {e}")
    st.stop()

if not rows:
    st.info(
        "No cluster assignments yet. Upload data first or run "
        "`POST /clusters/recompute`."
    )
    st.stop()

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── Summary ────────────────────────────────────────────────────────────────

counts = df["cluster_label"].value_counts().reset_index()
counts.columns = ["day_type", "count"]

c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("Day-type breakdown")
    st.dataframe(counts, use_container_width=True, hide_index=True)
with c2:
    fig = px.pie(counts, names="day_type", values="count", title="Days by archetype")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Timeline strip ─────────────────────────────────────────────────────────

st.subheader("Day-type timeline")
fig = px.scatter(
    df,
    x="date",
    y="cluster_label",
    color="cluster_label",
    title="Cluster assignments over time",
)
fig.update_traces(marker=dict(size=12))
fig.update_layout(showlegend=False, height=400)
st.plotly_chart(fig, use_container_width=True)

# ─── Action ─────────────────────────────────────────────────────────────────

with st.expander("Re-fit clusters"):
    n = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
    if st.button("Recompute"):
        with st.spinner("Re-fitting K-means..."):
            try:
                resp = api.post_clusters_recompute(n_clusters=n)
                st.success(resp.get("message", "Done."))
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Recompute failed: {e}")

with st.expander("Raw assignments"):
    st.dataframe(df, use_container_width=True)
