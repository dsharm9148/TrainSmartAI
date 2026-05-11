import requests
import streamlit as st

from frontend.lib import api

st.set_page_config(page_title="Chat | TrainSmartAI", layout="wide")
st.title("Chat with your data")
st.caption(
    "Ask anything about your health data. Answers are grounded in the indexed "
    "summaries via local Ollama models — nothing leaves your machine."
)

# ─── Sidebar: session controls ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Session")

    if st.button("New session"):
        st.session_state.pop("session_id", None)
        st.session_state["messages"] = []
        st.rerun()

    try:
        sessions = api.list_chat_sessions()
    except requests.RequestException as e:
        sessions = []
        st.caption(f"Could not list sessions: {e}")

    if sessions:
        options = {
            f"{s['session_id'][:8]} — {s['started_at'][:16]}": s["session_id"]
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

# ─── Render history ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input box ──────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask about your sleep, workouts, readiness…")
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
