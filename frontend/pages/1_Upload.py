import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Upload | TrainSmartAI", layout="wide")
st.title("Upload Apple Health Export")

st.markdown(
    """
    Upload the **`export.xml`** file from your Apple Health export. The ingestion
    pipeline parses every record, deduplicates against what's already in the
    database, and rebuilds daily features, readiness scores, weekly summaries,
    insights, and recommendations end-to-end.

    Re-uploading the same file is safe — duplicates are silently skipped.
    """
)

uploaded = st.file_uploader("Apple Health export.xml", type=["xml"])

if uploaded is not None:
    if st.button("Upload and process", type="primary"):
        with st.spinner("Ingesting and running the pipeline — this can take a minute on larger exports..."):
            try:
                result = api.upload_xml(uploaded.name, uploaded.getvalue())
            except requests.HTTPError as e:
                st.error(f"Upload failed: {e.response.text}")
            except requests.RequestException as e:
                st.error(f"Could not reach API at {api.BASE_URL}: {e}")
            else:
                st.success(result.get("message", "Upload complete"))

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Parsed", result["parsed"])
                c2.metric("Inserted", result["inserted"])
                c3.metric("Filtered", result["filtered"])
                c4.metric("Days computed", result["days_computed"])
                c5.metric("Weeks computed", result["weeks_computed"])

                with st.expander("Records by type"):
                    st.json(result.get("by_type", {}))

                st.info(
                    "Next: build the RAG index so the chat assistant can use your "
                    "data. From a terminal:\n\n"
                    "```bash\npython -c \"from backend.db.session import SessionLocal; "
                    "from backend.rag.indexer import index_health_data; "
                    "print(index_health_data(SessionLocal()), 'docs indexed')\"\n```"
                )
