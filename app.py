# app.py - Streamlit UI for SEC RAG Q&A
import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from ingest import ingest_pdf
from chain import ask, reset_conversation

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Annual Report Q&A",
    page_icon="📊",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Annual Report Q&A")
    st.caption("Upload any company annual report and ask questions in plain English.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        # Save uploaded file temporarily to disk (ingest_pdf needs a path)
        temp_path = f"data/{uploaded_file.name}"
        os.makedirs("data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ingest with a spinner — shows progress to user
        with st.spinner("Ingesting document... (first time takes ~30s)"):
            collection_id = ingest_pdf(temp_path)

        st.session_state["collection_id"] = collection_id
        st.session_state["doc_name"] = uploaded_file.name
        st.success(f"Ready — {uploaded_file.name}")

    # Show suggested questions only after upload
    if "collection_id" in st.session_state:
        st.divider()
        st.caption("SUGGESTED QUESTIONS")
        suggestions = [
            "What was the revenue in 2023?",
            "What are the key risks?",
            "What is the company strategy?",
            "What was the net income?",
            "How did margins change vs last year?",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state["prefill"] = s

    # Reset button
    if "messages" in st.session_state and st.session_state["messages"]:
        st.divider()
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            reset_conversation()
            st.rerun()

# ── Main chat area ────────────────────────────────────────────
if "collection_id" not in st.session_state:
    # Empty state — no document uploaded yet
    st.markdown("## 👈 Upload an annual report PDF to get started")
    st.markdown(
        "This tool lets you ask questions about any company's annual report "
        "in plain English. Try uploading Schneider Electric, TCS, or Apple's report."
    )
else:
    st.subheader(f"Chatting with: {st.session_state['doc_name']}")

    # Initialise message history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render existing chat messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"📄 Sources: Pages {msg['sources']}")

    # Handle prefilled question from sidebar button click
    prefill = st.session_state.pop("prefill", None)

    # Chat input
    user_input = st.chat_input("Ask a question about the report...") or prefill

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(user_input, st.session_state["collection_id"])
            st.markdown(result["answer"])
            st.caption(f"📄 Sources: Pages {result['sources']}")

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })