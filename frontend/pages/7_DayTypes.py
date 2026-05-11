import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import empty_state, hero, inject_global_css

st.set_page_config(page_title="Day Types | TrainSmartAI", layout="wide")
inject_global_css()

hero(
    "Day Types",
    "K-means clusters every day by activity, sleep, resting HR, and workout load — each centroid gets an automatic archetype label.",
)

try:
    rows = api.get_clusters()
except requests.RequestException as e:
    st.error(f"Failed to fetch clusters: {e}")
    st.stop()

if not rows:
    empty_state(
        "No cluster assignments yet",
        "Upload data first or trigger clustering manually.",
        "Action: `POST /clusters/recompute`",
    )
    st.stop()

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ─── Summary ────────────────────────────────────────────────────────────────

counts = df["cluster_label"].value_counts().reset_index()
counts.columns = ["day_type", "count"]
counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1).astype(str) + "%"

c1, c2 = st.columns([2, 3])
with c1:
    with st.container(border=True):
        st.markdown("**Day-type breakdown**")
        st.dataframe(counts, use_container_width=True, hide_index=True)
with c2:
    fig = px.pie(counts, names="day_type", values="count", template="simple_white")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ─── Timeline ───────────────────────────────────────────────────────────────

st.subheader("Day-type timeline")
fig = px.scatter(
    df,
    x="date",
    y="cluster_label",
    color="cluster_label",
    template="simple_white",
)
fig.update_traces(marker=dict(size=12, line=dict(width=0)))
fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

# ─── Re-fit ─────────────────────────────────────────────────────────────────

with st.expander("Re-fit clusters"):
    n = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
    if st.button("Recompute", type="primary"):
        with st.spinner("Re-fitting K-means..."):
            try:
                resp = api.post_clusters_recompute(n_clusters=n)
                st.success(resp.get("message", "Done."))
                st.rerun()
            except requests.RequestException as e:
                st.error(f"Recompute failed: {e}")

with st.expander("Raw assignments"):
    st.dataframe(df, use_container_width=True)
