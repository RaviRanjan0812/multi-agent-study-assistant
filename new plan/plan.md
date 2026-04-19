# AI Study Assistant — Consolidated build plan (`plan.md`)

> **Time budget:** 6 hours | **Team:** 3 | **Framework:** LangGraph  
> **Source:** Merged from `HACKATHON_PLAN (1).md` with **document isolation**, **ephemeral vector memory**, and **Streamlit session** contract added.

---

## How this file relates to `HACKATHON_PLAN (1).md`

- **`plan.md`** (this file): single place to **implement from** — includes hackathon scope **plus** multi-document safety and RAM lifecycle.
- **`HACKATHON_PLAN (1).md`**: original long reference; keep in repo for history if you like.

---

## Project summary

A **multi-agent AI study assistant** powered by LangGraph: specialized agents collaborate so a query like *"Teach me photosynthesis and test me"* yields an explanation **and** a quiz.

**Differentiators:**

- LangGraph state machine with conditional routing (not flat if/elif)
- Two-stage retrieval: BM25 + FAISS → RRF fusion → cross-encoder reranking
- Separate fast vs reasoning paths by intent
- Different LLM sizes per agent (cost/speed)

---

## Document scope, isolation, and memory lifecycle

### Problem

Indexing **multiple PDFs at once** (or leaving old vectors in RAM) causes **cross-topic contamination** (biology + history in one retriever) and **RAM growth** (risk of OOM on a laptop demo).

### Hackathon approach (default): session-scoped corpus

- **One active document** (or one rebuilt corpus) per session.
- On **new upload** or **Clear memory**: drop `st.session_state` references to the vector store **and** BM25 store, clear chat if desired, so old embeddings become unreachable and can be garbage-collected.
- **FAISS + BM25 must always be rebuilt from the same `chunks` list.** Never refresh only one side.

### Enterprise direction (pitch only, not 6h build)

- One vector backend, **chunk metadata** (e.g. `filename`, `user_id`), **filtered retrieval** or **tenant namespaces** (Pinecone, Qdrant, etc.).
- Planner or a router supplies filters for retrieval. Out of scope for the hackathon code path above.

### Judge line (ephemeral vs production)

> For this prototype, our FAISS index is **ephemeral**: it lives in **process RAM** for the active Streamlit session. When the session ends or the user uploads a new document (or clears memory), we **rebuild** the index so vectors are not retained indefinitely — avoiding OOM and cross-document leakage in the demo. In production we would use a **persistent vector database** with **strict partitioning** (e.g. per-`user_id`) for privacy and recall across sessions.

### Streamlit session contract (implement `app.py` this way)

- **Sidebar:** file uploader (PDF), **Clear memory & start over** (sets `vs`, `bm25_store`, `messages`, and `current_file` as needed).
- **Guard:** only re-ingest when `current_file` differs from the active upload’s identity (see pitfall: same filename, different bytes).
- **No `@st.cache_resource` for the pair `(FAISS, BM25)`** when using per-upload indexing — cache would fight session resets.

---

## System architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│         Planner Agent           │
│   (LangGraph Conditional Edge)  │
└────────────────┬────────────────┘
                 │
     ┌───────────┴───────────┐
     │ (quick_question)      │ (learn / quiz / learn_and_test)
     ▼                       ▼
┌──────────────┐    ┌─────────────────────────────────────┐
│ Fast Response│    │          Research Agent              │
│ Agent        │    │  Stage 1a: BM25 keyword search       │
│ llama3-8b    │    │  Stage 1b: FAISS vector search       │
└──────┬───────┘    │  RRF Fusion (top 10 combined)        │
       │            │  Stage 2: Cross-Encoder rerank → 3   │
       │            └──────────────────┬──────────────────┘
       │                               │
       │              ┌────────────────┴────────────────┐
       │              ▼                                  ▼
       │    ┌──────────────────┐             ┌──────────────────┐
       │    │ Explanation Agent│             │   Quiz Agent     │
       │    │  llama3-70b      │             │   llama3-70b     │
       │    └────────┬─────────┘             └────────┬─────────┘
       │             │                                 │
       │             └──────────────┬──────────────────┘
       │                            ▼
       │                 ┌──────────────────┐
       │                 │ Synthesizer Node │
       │                 └────────┬─────────┘
       │                          │
       └──────────────────────────┘
                                  │
                                  ▼
                     Final Output (Streamlit UI)
```

---

## Agent roles

| Agent | Role | Model | What it does |
|-------|------|-------|--------------|
| **Planner Agent** | Coordinator | llama3-70b | Classifies intent, sets LangGraph routing |
| **Research Agent** | RAG + rerank | — | BM25 + FAISS → RRF → cross-encoder |
| **Explanation Agent** | Tutor | llama3-70b | Structures the concept |
| **Quiz Agent** | Examiner | llama3-70b | MCQs / short answers |
| **Fast Response Agent** | Quick help | llama3-8b | No RAG, low latency |
| **Synthesizer Node** | Combiner | — | Merges explanation + quiz |

---

## Two-stage retrieval pipeline

```
User Query
    │
    ├──► BM25 Search (keyword match) ──► Top 10 docs
    │
    ├──► FAISS Search (semantic match) ──► Top 10 docs
    │
    ▼
RRF Fusion (Reciprocal Rank Fusion)
    │  RRF(d) = Σ 1/(k + rank_i(d))  where k=60
    ▼
Cross-Encoder Reranker
    │  Top 3 chunks → LLM context
    ▼
LangGraph state → agents
```

**Judge talking points:** BM25 for exact terms, FAISS for paraphrase, RRF without hand-tuned blend weights, cross-encoder cuts noise before the LLM.

---

## Full workflow (step by step)

1. **(Upload path)** User uploads a PDF (sidebar) → save temp file → **chunk once** → build **FAISS + BM25** from those chunks → store both in `st.session_state`. If they upload another file or click **Clear**, rebuild or clear as above.
2. User types a query in the Streamlit chat UI.
3. **Planner** classifies: `learn_only` | `quiz_only` | `learn_and_test` | `quick_question`.
4. LangGraph routes via conditional edges.
5. **Research** runs BM25 + FAISS → RRF → cross-encoder (skip or stub context for `quick_question` per graph design).
6. Context in state → explanation and/or quiz nodes.
7. **Synthesizer** merges outputs.
8. Final message shown in chat; history in `st.session_state.messages`.

---

## Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Orchestration | LangGraph | State + routing |
| LLM (heavy) | Groq llama3-70b-8192 | Explain / quiz / planner |
| LLM (fast) | Groq llama3-8b-8192 | Quick path |
| Vectors | FAISS | In-memory, no hosted DB for hackathon |
| Keywords | rank_bm25 | Complements FAISS |
| Fusion | RRF | Simple, strong baseline |
| Reranker | cross-encoder/ms-marco-MiniLM-L-4-v2 | CPU-friendly |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | Local |
| UI | Streamlit | Fast demo surface |

---

## Dependencies (`requirements.txt`)

```
langchain==0.2.0
langchain-groq==0.1.3
langchain-community==0.2.0
langgraph==0.0.65
faiss-cpu==1.8.0
sentence-transformers==2.7.0
rank-bm25==0.2.2
streamlit==1.35.0
python-dotenv==1.0.1
pypdf==4.2.0
```

```bash
pip install -r requirements.txt
```

---

## Project structure

```
ai-study-assistant/
├── app.py
├── requirements.txt
├── .env
├── agents/
│   ├── planner_agent.py
│   ├── explanation_agent.py
│   ├── quiz_agent.py
│   └── fast_response_agent.py
├── memory/
│   ├── vector_store.py
│   ├── bm25_store.py
│   ├── retriever.py
│   └── sample_notes/          # optional static samples for dev
└── graph/
    └── study_graph.py
```

---

## Code sketches — core files

### `.env`

```
GROQ_API_KEY=your_groq_api_key_here
```

https://console.groq.com

---

### `memory/vector_store.py` (directory **or** single session file)

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_documents(docs_path="memory/sample_notes"):
    """Load all supported files from a folder (dev / static demo)."""
    documents = []
    for filename in os.listdir(docs_path):
        path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def load_documents_from_file(file_path: str):
    """Load one file — use for Streamlit upload → temp path (session scope)."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def faiss_search(query: str, vector_store, k=10):
    return vector_store.similarity_search(query, k=k)
```

**Session helper (call from `app.py`):** `docs = load_documents_from_file(path)` → `chunks = split_documents(docs)` → `vs = build_vector_store(chunks)` and `BM25Store(chunks)` together.

---

### `memory/bm25_store.py`

```python
from rank_bm25 import BM25Okapi

class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k=10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]
```

---

### `memory/retriever.py`

(Unchanged logic from hackathon plan: `rrf_fusion`, `cross_encoder_rerank`, `retrieve(query, vector_store, bm25_store)`.)

---

### `agents/planner_agent.py`, `explanation_agent.py`, `quiz_agent.py`, `fast_response_agent.py`

(Same as in `HACKATHON_PLAN (1).md` — copy from there when implementing.)

---

### `graph/study_graph.py`

(Same `AgentState`, nodes, `build_graph`, `study_graph` as hackathon plan; pass `vector_store` and `bm25_store` from session state at `invoke()`.)

---

### `app.py` (session-scoped stores + sidebar)

```python
import streamlit as st
from dotenv import load_dotenv
from memory.vector_store import load_documents_from_file, split_documents, build_vector_store
from memory.bm25_store import BM25Store
from graph.study_graph import study_graph

load_dotenv()

st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AI Study Assistant")
st.caption("Multi-Agent AI — LangGraph + BM25 + FAISS + Cross-Encoder")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Study materials")
    uploaded_file = st.file_uploader("Upload a PDF to study", type=["pdf"])
    if st.button("Clear memory & start over"):
        st.session_state.vs = None
        st.session_state.bm25_store = None
        st.session_state.messages = []
        st.session_state.current_file = None
        st.success("Session memory cleared.")

def ingest_uploaded_pdf(uploaded_file) -> None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    docs = load_documents_from_file(tmp_path)
    chunks = split_documents(docs)
    st.session_state.vs = build_vector_store(chunks)
    st.session_state.bm25_store = BM25Store(chunks)
    st.session_state.current_file = getattr(uploaded_file, "file_id", None) or (
        uploaded_file.name, uploaded_file.size
    )

if uploaded_file is not None:
    key = getattr(uploaded_file, "file_id", None) or (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("current_file") != key:
        with st.spinner("Processing document and building vector memory..."):
            ingest_uploaded_pdf(uploaded_file)
        st.success(f"Loaded {uploaded_file.name} into active memory.")
else:
    st.info("Upload a document to begin a deep-dive study session.")
    st.session_state.vs = None
    st.session_state.bm25_store = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("E.g. 'Teach me photosynthesis and test me'"):
    if st.session_state.vs is None or st.session_state.bm25_store is None:
        st.warning("Upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Agents working..."):
                result = study_graph.invoke({
                    "query": query,
                    "intent": "",
                    "context": "",
                    "explanation": "",
                    "quiz": "",
                    "final_output": "",
                    "vector_store": st.session_state.vs,
                    "bm25_store": st.session_state.bm25_store,
                })
                final = result["final_output"]
                st.markdown(final)
        st.session_state.messages.append({"role": "assistant", "content": final})
```

---

## 6-hour build timeline (updated)

| Time | Person 1 | Person 2 | Person 3 |
|------|----------|----------|----------|
| **Hour 1** | Setup: venv, `.env`, tree, deps | `vector_store.py` + `bm25_store.py` (+ single-file loader) | `planner_agent.py`, test intents |
| **Hour 2** | `retriever.py` (RRF + cross-encoder) | `explanation_agent.py` | `quiz_agent.py` |
| **Hour 3** | `fast_response_agent.py` | `study_graph.py` | **`app.py`**: sidebar upload, clear, session `vs` + `bm25_store`, wire graph |
| **Hour 4** | E2E all intents | Empty store, bad intent, **FAISS/BM25 mismatch** bugs | UI polish |
| **Hour 5** | Sample notes / PDF tests | Edge cases (re-upload, clear) | Deploy / localhost |
| **Hour 6** | Buffer / debug | Demo script | 2-minute pitch |

---

## Quick start (abbrev.)

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
# GROQ_API_KEY in .env
streamlit run app.py
```

---

## Demo queries for judges

| Query | Intent | What judges see |
|-------|--------|-----------------|
| `"What is ATP?"` | `quick_question` | Fast path, small model |
| `"Teach me the French Revolution"` | `learn_only` | Full RAG pipeline |
| `"Teach me Newton's laws and test me"` | `learn_and_test` | Explain + quiz |

---

## Pitch points (add to slide)

- Two retrievers + RRF + cross-encoder before the LLM.
- LangGraph orchestration, dual LLM sizes.
- **Ephemeral, session-scoped vectors** for a clean demo; **production = persistent DB + namespaces**.

---

## Pitfalls to avoid

- **FAISS crashes if no docs** — block chat until upload succeeds or ship one default doc.
- **Never desync FAISS and BM25** — same `chunks` for both on every rebuild.
- **Cross-encoder model size** — keep `ms-marco-MiniLM-L-4-v2` on CPU for hackathon.
- **LangGraph `invoke`** — pass all required state keys.
- **Groq rate limits** — fallback to smaller model on planner if needed.
- **OOM** — do not accumulate unlimited uploads; clear or single active doc.
- **Same filename, different file** — key ingest on `file_id` or `(name, size)` or content hash, not name alone.

---

## Full code blocks for agents + graph + retriever

The hackathon document **`HACKATHON_PLAN (1).md`** contains the complete, copy-paste versions of:

- `memory/retriever.py`
- all files under `agents/`
- `graph/study_graph.py`
- the original `app.py` (static `load_documents` + `@st.cache_resource`) for comparison

Use **`plan.md`** for behavior and **`HACKATHON_PLAN (1).md`** for line-complete snippets where this file points to “same as hackathon.”
