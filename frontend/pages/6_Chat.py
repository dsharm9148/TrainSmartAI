import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests
import streamlit as st

from frontend.lib import api
from frontend.lib.ui import hero, inject_global_css, pill

st.set_page_config(page_title="Chat | TrainSmartAI", layout="wide")
inject_global_css()

hero(
    "Chat with your data",
    "Ask anything about your sleep, workouts, readiness, or trends. Answers are grounded in your indexed health summaries via a local Ollama model — nothing leaves your machine.",
)

# ─── Sidebar: session controls ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Conversation")

    if st.button("New session", use_container_width=True):
        st.session_state.pop("session_id", None)
        st.session_state["messages"] = []
        st.rerun()

    try:
        sessions = api.list_chat_sessions()
    except requests.RequestException as e:
        sessions = []
        st.caption(f"Could not list sessions: {e}")

    if sessions:
        st.caption(f"{len(sessions)} past session(s)")
        options = {
            f"{s['session_id'][:8]} · {s['started_at'][:16]}": s["session_id"]
            for s in sessions
        }
        labels = ["(new session)"] + list(options)
        idx = 0
        current = st.session_state.get("session_id")
        if current is not None:
            for i, sid in enumerate(options.values(), start=1):
                if sid == current:
                    idx = i
                    break

        choice = st.selectbox("Resume", labels, index=idx)
        if choice != "(new session)":
            sid = options[choice]
            if sid != st.session_state.get("session_id"):
                st.session_state["session_id"] = sid
                try:
                    history = api.get_chat_messages(sid)
                    st.session_state["messages"] = [
                        {"role": h["role"], "content": h["content"]} for h in history
                    ]
                except requests.RequestException as e:
                    st.error(f"Failed to load history: {e}")
                st.rerun()

    st.divider()
    st.caption("Try asking:")
    for sample in [
        "How did I sleep last week?",
        "Which day was my best workout?",
        "Why is my readiness low recently?",
        "Should I train today or rest?",
    ]:
        if st.button(sample, key=f"sample_{sample}", use_container_width=True):
            st.session_state["_prefill"] = sample

# ─── Status header ──────────────────────────────────────────────────────────

sid = st.session_state.get("session_id")
status_a, status_b = st.columns([1, 5])
with status_a:
    if sid:
        st.markdown(pill("Active session", "green"), unsafe_allow_html=True)
        st.caption(f"`{sid[:8]}…`")
    else:
        st.markdown(pill("New session", "slate"), unsafe_allow_html=True)
with status_b:
    st.caption("First reply takes ~10 s on M-series Mac while the LLM warms up. Subsequent replies are fast.")

st.divider()

# ─── Render history ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input ──────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("_prefill", None)
prompt = st.chat_input("Ask about your sleep, workouts, readiness…") or prefill
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking — local LLM, first reply takes ~10s..."):
            try:
                resp = api.post_chat(prompt, session_id=st.session_state.get("session_id"))
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                st.stop()
            except requests.RequestException as e:
                st.error(f"Could not reach API: {e}")
                st.stop()

        st.session_state["session_id"] = resp["session_id"]
        answer = resp["response"]
        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
