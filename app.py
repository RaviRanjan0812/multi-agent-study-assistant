"""Streamlit UI + LangGraph entry point."""

import os
import tempfile

import groq
import streamlit as st
from dotenv import load_dotenv

from graph.study_graph import study_graph
from memory.bm25_store import BM25Store
from memory.vector_store import build_vector_store, load_documents_from_file, split_documents

_ROOT = os.path.abspath(os.path.dirname(__file__))
_ENV_PATH = os.path.join(_ROOT, ".env")
load_dotenv(_ENV_PATH, override=True)


def _groq_key_configured() -> bool:
    k = (os.getenv("GROQ_API_KEY") or "").strip().strip("\ufeff")
    return bool(k) and "your_groq_api_key" not in k.lower()


st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AI Study Assistant")
st.caption("LangGraph Multi-Agent · BM25 + FAISS + Cross-Encoder RAG")

if not _groq_key_configured():
    st.warning(
        "Add a valid `GROQ_API_KEY` in `.env` at the project root (no quotes, no spaces). "
        "Create a key at https://console.groq.com — then save `.env` and use **Rerun** in the Streamlit menu."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "bm25_store" not in st.session_state:
    st.session_state.bm25_store = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.sidebar:
    st.header("📚 Study Materials")
    uploaded_file = st.file_uploader("Upload a PDF to study", type=["pdf"])

    if st.button("🗑️ Clear Memory & Start Over"):
        st.session_state.vs = None
        st.session_state.bm25_store = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.success("Session memory cleared.")

    st.markdown("---")
    st.markdown("**Session status:**")
    if st.session_state.vs is not None:
        st.success("Document loaded ✓")
    else:
        st.warning("No document loaded")


def ingest_uploaded_pdf(uploaded_file) -> None:
    """Save upload to temp file → load → chunk → FAISS + BM25 from same chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        docs = load_documents_from_file(tmp_path)
    finally:
        os.remove(tmp_path)

    chunks = split_documents(docs)
    st.session_state.vs = build_vector_store(chunks)
    st.session_state.bm25_store = BM25Store(chunks)
    st.session_state.current_file = getattr(uploaded_file, "file_id", None) or (
        uploaded_file.name,
        uploaded_file.size,
    )


if uploaded_file is not None:
    file_key = getattr(uploaded_file, "file_id", None) or (
        uploaded_file.name,
        uploaded_file.size,
    )
    if st.session_state.current_file != file_key:
        with st.spinner("Building vector memory and cross-encoder index..."):
            ingest_uploaded_pdf(uploaded_file)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
else:
    if st.session_state.current_file is not None:
        st.session_state.vs = None
        st.session_state.bm25_store = None
        st.session_state.current_file = None
    st.sidebar.info("Upload a PDF for deep RAG, or just chat with the Fast Agent!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("E.g. 'Teach me photosynthesis and test me'"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Agents orchestrating..."):
            final = ""
            try:
                result = study_graph.invoke(
                    {
                        "query": query,
                        "intent": "",
                        "context": "",
                        "explanation": "",
                        "quiz": "",
                        "final_output": "",
                        "vector_store": st.session_state.vs,
                        "bm25_store": st.session_state.bm25_store,
                    }
                )
                final = result["final_output"]
            except groq.AuthenticationError:
                final = (
                    "**Invalid Groq API key (401).** Fix `.env`: use a fresh key from "
                    "[Groq Console](https://console.groq.com), one line `GROQ_API_KEY=gsk_...`, "
                    "no quotes or spaces, save the file, then **Rerun** the app."
                )
                st.error(final)
            except Exception as e:
                final = f"**Unexpected error:** {e}"
                st.error(final)
            else:
                st.markdown(final)

    if final:
        st.session_state.messages.append({"role": "assistant", "content": final})
