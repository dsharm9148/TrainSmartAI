import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import empty_state, hero, inject_global_css

st.set_page_config(page_title="Upload | TrainSmartAI", layout="wide")
inject_global_css()

hero(
    "Upload Apple Health export",
    "Ingestion runs end-to-end: parse → clean → upsert → daily features → readiness → weekly summaries → insights → recommendations → clustering.",
)

c_left, c_right = st.columns([2, 1])

with c_left:
    uploaded = st.file_uploader(
        "Apple Health export.xml",
        type=["xml"],
        help="On iPhone: Health app → profile → Export All Health Data. Unzip and use the export.xml inside.",
    )

    if uploaded is not None:
        st.caption(f"Selected: **{uploaded.name}** ({uploaded.size / (1024 * 1024):.1f} MB)")
        if st.button("Upload and process", type="primary", use_container_width=True):
            with st.spinner("Ingesting and running the pipeline — this can take a minute for large exports..."):
                try:
                    result = api.upload_xml(uploaded.name, uploaded.getvalue())
                except requests.HTTPError as e:
                    st.error(f"Upload failed: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Could not reach API at {api.BASE_URL}: {e}")
                else:
                    st.success(result.get("message", "Upload complete"))
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Parsed", f"{result['parsed']:,}")
                    m2.metric("Inserted", f"{result['inserted']:,}")
                    m3.metric("Filtered", result["filtered"])
                    m4.metric("Days computed", result["days_computed"])
                    m5.metric("Weeks computed", result["weeks_computed"])

                    with st.expander("Records by type"):
                        st.json(result.get("by_type", {}))

with c_right:
    with st.container(border=True):
        st.markdown("**Re-uploading is safe**")
        st.caption(
            "Duplicates are skipped via a `(record_type, source, start, end)` unique "
            "constraint, so you can refresh the dataset as often as you like."
        )

    with st.container(border=True):
        st.markdown("**Next step after upload**")
        st.caption("Build the RAG index so the chat assistant can use your data:")
        st.code(
            'python -c "from backend.db.session import SessionLocal; '
            "from backend.rag.indexer import index_health_data; "
            "print(index_health_data(SessionLocal()), 'docs indexed')\"",
            language="bash",
        )

    with st.container(border=True):
        st.markdown("**No watch? No problem.**")
        st.caption("Generate 90 days of realistic synthetic data:")
        st.code("python data/synthetic/generate_sample.py", language="bash")

if uploaded is None:
    st.divider()
    empty_state(
        "No file selected",
        "Drop your `export.xml` above to begin. The pipeline auto-runs daily features, readiness, weekly summaries, insights, recommendations, and clustering.",
        "Tip: re-uploading the same file is a no-op — perfect for refreshing the demo.",
    )
