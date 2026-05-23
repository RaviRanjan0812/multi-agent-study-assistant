# 🎓 AI Study Assistant — Multi-Agent RAG System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B6B?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203-F55036?style=for-the-badge&logo=meta&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-009688?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

> **An intelligent, multi-agent AI-powered study companion that explains topics, retrieves relevant context from your own uploaded PDFs, and tests your knowledge — all in one seamless conversational interface.**

[Quick Start](#-quick-start) • [Architecture](#-system-architecture) • [Features](#-features) • [Installation](#-installation) • [Configuration](#-configuration) • [Usage](#-usage) • [Team](#-team)

</div>

---

## About the Project

**AI Study Assistant** is built for the **HCL Tech Hackathon**. It is a production-quality, intelligent study companion powered by a **multi-agent LangGraph workflow** combined with a **hybrid RAG (Retrieval-Augmented Generation)** pipeline.

The system understands *what* you want — learn, be quizzed, or just ask a quick question — and dynamically routes your query through specialized AI agents to deliver structured explanations and MCQ-based quizzes, all grounded in your own uploaded study material.

---

## Features

### Smart Intent Classification
The **Planner Agent** (LLaMA 3.3-70B via Groq) classifies every query into one of five intents — `learn_and_test`, `learn_only`, `quiz_only`, `quick_question`, or `unclear_intent` — and routes execution dynamically through the LangGraph workflow.

### Hybrid RAG Pipeline
Upload **any PDF** and the system builds a full hybrid retrieval index in seconds:
- **FAISS** — dense semantic vector search using `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **BM25** — sparse keyword search via `rank-bm25`
- **Reciprocal Rank Fusion (RRF)** — merges both ranked lists without hand-tuned blend weights

### Cross-Encoder Reranking with Lost-in-the-Middle Fix
After RRF fusion, a **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-4-v2`) precisely scores each (query, chunk) pair. The top 3 results are then **reordered** — best chunk first, second-best last — to combat the "Lost-in-the-Middle" LLM attention problem.

### Dual-Speed LLM Backends
- **LLaMA 3.3-70B Versatile** — deep explanations, MCQ generation, and intent classification
- **LLaMA 3.1-8B Instant** — handles fast-path responses for greetings and simple queries
- Both served via **Groq API** for ultra-low latency inference

### Structured MCQ Generation
The **Quiz Agent** generates exactly **3 formatted multiple-choice questions** per topic with lettered options and instant correct-answer explanations — whether from your uploaded notes or the LLM's own general knowledge.

### Security Guardrails
Built-in **jailbreak detection** in the Planner Agent catches prompt injection, instruction-override attacks, and system prompt extraction — redirecting them politely without breaking the UX.

### LangGraph Stateful Orchestration
The entire multi-agent workflow is a **compiled LangGraph `StateGraph`** with a typed `AgentState`. Conditional edges dynamically include or skip explanation and quiz nodes based on the classified intent.

### PDF-First Study Mode
Upload lecture notes, textbooks, or research papers from the Streamlit sidebar. The system ingests → parses → chunks (500 tokens / 50 overlap) → embeds → indexes the content. Clear memory with one click to start a fresh session.

---

## System Architecture

```
+-----------------------------------------------------------------------------------+
|                         STREAMLIT UI  (app.py)                                    |
|   PDF Upload => FAISS + BM25 Indexing | Chat Interface | Session State Management |
+-----------------------------+-----------------------------------------------------+
                              |  query + vector_store + bm25_store
                              v
+-----------------------------------------------------------------------------------+
|                    LANGGRAPH WORKFLOW  (graph/study_graph.py)                     |
|                                                                                   |
|  +-----------+  intent   +------------------------------------------------+       |
|  |  PLANNER  +---------->+            CONDITIONAL ROUTER                  |       |
|  | (70B LLM) |          |  learn_and_test / learn_only / quiz_only => RESEARCH   |
|  +-----------+          |  quick_question / unclear_intent => FAST PATH   |       |
|                         +------------------------------------------------+       |
|                                                                                   |
|  +-------------------------------------------------------------------------+      |
|  |  RESEARCH NODE  (memory/retriever.py)                                   |      |
|  |  BM25(k=10) --+                                                         |      |
|  |               +---> RRF Fusion (top 10) --> Cross-Encoder Rerank (top 3)|      |
|  |  FAISS(k=10) -+                   + Lost-in-the-Middle Reorder          |      |
|  +-------------------------------------------------------------------------+      |
|                                                                                   |
|  +----------------+  +------------+  +------------------+                        |
|  |  EXPLANATION   +->+    QUIZ    +->+   SYNTHESIZER    +-> final_output         |
|  |  (70B Tutor)   |  | (70B MCQ) |  |  (merge output)  |                        |
|  +----------------+  +------------+  +------------------+                        |
|                                                                                   |
|  +----------------+                                                               |
|  |  FAST RESPONSE +-------------------------------------------------> END         |
|  |  (8B Instant)  |                                                               |
|  +----------------+                                                               |
+-----------------------------------------------------------------------------------+
```

### Agent Routing Table

| User Intent | LangGraph Route | Agents Activated |
|---|---|---|
| `learn_and_test` | Research → Explain → Quiz → Synthesize | Planner, Research, Explanation, Quiz, Synthesizer |
| `learn_only` | Research → Explain → Synthesize | Planner, Research, Explanation, Synthesizer |
| `quiz_only` | Research → Quiz → Synthesize | Planner, Research, Quiz, Synthesizer |
| `quick_question` | Fast Path | Planner, Fast Response |
| `unclear_intent` | Fast Path | Planner, Fast Response |
| `malicious_intent` | Fast Path (polite decline) | Planner, Fast Response |

### RAG Pipeline — Step by Step

```
PDF Upload
    |
    v
PyPDFLoader => Raw Documents
    |
    v
RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
    |
    +---> HuggingFace Embeddings (all-MiniLM-L6-v2) ---> FAISS Index
    |
    +---> BM25Store (rank-bm25)

At Query Time:
    Query ---> FAISS.similarity_search(k=10) -------------+
    Query ---> BM25Store.search(k=10) -------------------+
                                                          v
                                       RRF Fusion => top 10 candidates
                                                          |
                                                          v
                                       Cross-Encoder Rerank => top 3
                                                          |
                                                          v
                                       Lost-in-Middle Reorder
                                                          |
                                                          v
                                            Context String => LLM
```

---

## Project Structure

```
multi-agent-study-assistant/
|
+-- app.py                        # Streamlit UI, session management, PDF ingestion
+-- requirements.txt              # All Python dependencies (flexible version pins)
+-- .env                          # API keys — NOT committed to git
+-- .gitignore                    # Ignores .env, .venv, __pycache__, etc.
|
+-- agents/                       # All specialized LangGraph agent modules
|   +-- __init__.py
|   +-- groq_llms.py              # Groq LLM factory (70B + 8B, dynamic .env reload)
|   +-- planner_agent.py          # Intent classifier + security guardrail prompt
|   +-- explanation_agent.py      # Tutor agent (structured 3-part explanation)
|   +-- quiz_agent.py             # MCQ generator (3 questions, formatted output)
|   +-- fast_response_agent.py    # Fast-path for greetings / off-topic queries
|
+-- graph/                        # LangGraph state machine definition
|   +-- __init__.py
|   +-- study_graph.py            # AgentState, nodes, conditional edges, compile()
|
+-- memory/                       # Full RAG pipeline
|   +-- __init__.py
|   +-- vector_store.py           # FAISS: load, split, embed, search
|   +-- bm25_store.py             # BM25 keyword index wrapper
|   +-- retriever.py              # RRF fusion + Cross-Encoder rerank + LitM reorder
|   +-- sample_notes/             # Sample PDFs/TXTs for development testing
|
+-- scripts/
    +-- verify_components.py      # Component sanity-check / smoke test script
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher |
| pip | 23.0 or higher |
| Git | Any recent version |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |

> No GPU required — the project runs entirely on CPU.

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-org>/multi-agent-study-assistant.git
cd multi-agent-study-assistant
```

---

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

**Activate it:**

```bash
# Windows — PowerShell
.\.venv\Scripts\Activate.ps1

# Windows — Command Prompt
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

You should see `(.venv)` at the beginning of your terminal prompt.

---

### Step 3 — Install All Dependencies

```bash
pip install -r requirements.txt
```

> **First-run note:** The HuggingFace embedding model (`all-MiniLM-L6-v2`, ~90 MB) and the cross-encoder (`ms-marco-MiniLM-L-4-v2`, ~70 MB) will be downloaded and cached at `~/.cache/huggingface/`. This only happens once.

---

## Configuration

### Setting Up Your Groq API Key

1. Go to [console.groq.com](https://console.groq.com) and sign up for a **free account**.
2. Navigate to **API Keys** → **Create API Key**.
3. Copy the key — it starts with `gsk_...`.
4. Create / open the `.env` file in the project root and add:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

> **Important `.env` Rules:**
> - No quotes around the key value — plain text only
> - No spaces before or after the `=`
> - Save the file in plain UTF-8 encoding (no BOM)

The app hot-reloads the key on every API request — no Streamlit restart needed. Use the **Rerun** option in the Streamlit menu after editing `.env`.

---

### Models Used

| Model ID | Role | Latency |
|---|---|---|
| `llama-3.3-70b-versatile` | Intent classification, explanations, MCQ generation | Moderate |
| `llama-3.1-8b-instant` | Fast path responses (greetings, off-topic) | Ultra-fast |

---

### Dependencies Overview

| Package | Purpose |
|---|---|
| `streamlit` | Web UI and chat interface |
| `langchain`, `langchain-core`, `langchain-groq` | LLM chains, prompt templates, Groq integration |
| `langgraph` | Multi-agent stateful workflow orchestration |
| `langchain-community` | FAISS, HuggingFace embeddings, PDF/text loaders |
| `langchain-text-splitters` | Recursive character text splitter |
| `faiss-cpu` | Dense vector similarity search index |
| `sentence-transformers` | Embedding model + Cross-Encoder reranker |
| `rank-bm25` | BM25 sparse keyword retrieval |
| `pypdf` | PDF parsing and text extraction |
| `python-dotenv` | `.env` file loading |
| `groq` | Groq Python SDK |

---

## Quick Start

After completing installation and configuration:

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Usage

### Without a PDF (General Knowledge Mode)

Type any educational question. The agents use the LLM's built-in knowledge.

| Example Query | Intent Detected | What You Get |
|---|---|---|
| `"Teach me photosynthesis"` | `learn_and_test` | Full explanation + 3 MCQs |
| `"Explain gravity, no quiz"` | `learn_only` | Explanation only |
| `"Quiz me on DNA replication"` | `quiz_only` | 3 MCQs directly |
| `"Hi"` or `"Hello"` | `quick_question` | Warm greeting + prompt |
| Random gibberish | `unclear_intent` | Polite redirection |
| Jailbreak attempt | `malicious_intent` | Polite decline |

---

### With a PDF (RAG Mode)

1. **Upload a PDF** using the sidebar file uploader (textbook, lecture notes, research paper).
2. Wait for the **"Document loaded ✓"** confirmation in the sidebar.
3. Ask any question — answers are **grounded in your document**.
4. Click **"Clear Memory & Start Over"** to remove the document and reset the session.

---

### Example Conversation

```
You:          "Teach me about the mitochondria and then quiz me"

Planner:      Detected intent: learn_and_test
Research:     Retrieved top 3 relevant chunks (or general knowledge)

AI Response:
## Explanation
1. Definition: The mitochondria is a membrane-bound organelle...
2. How it works: ATP synthesis via the electron transport chain...
3. Real-world analogy: Think of it as the power plant of the cell...

---

## Quiz
---
**Question 1:** What is the primary function of the mitochondria?
- A  Energy storage
- B  Protein synthesis
- C  ATP production
- D  DNA replication

Correct Answer: C — The mitochondria produces ATP through cellular respiration...
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `AuthenticationError (401)` | Fix your `.env` file — no quotes, no spaces. Get a fresh key from [console.groq.com](https://console.groq.com) |
| `GROQ_API_KEY is not set` | Make sure `.env` is in the project root (same folder as `app.py`) |
| Red underlines in IDE for `streamlit`/`dotenv` | Ensure `.venv` is activated and selected as the Python interpreter |
| Slow first startup | HuggingFace models are downloading (~160 MB total) — this is a one-time event |
| PDF not loading | Ensure the file is a valid, non-encrypted PDF. Try a different PDF if the issue persists |
| Model not found error | The Groq model may have changed. Check [Groq's model list](https://console.groq.com/docs/models) |

---

## How the Technology Stack Works Together

```
User types a query
        |
        v
Streamlit captures input and current session state (vector_store, bm25_store)
        |
        v
LangGraph invokes study_graph with the full AgentState
        |
        v
Planner Agent (LLaMA 3.3-70B) classifies intent via Groq API
        |
        +--[quick/unclear/malicious]--> Fast Response Agent (LLaMA 3.1-8B) --> END
        |
        +--[learn/quiz/both]---------> Research Node
                                            |
                                            v
                                    BM25 + FAISS retrieval
                                    RRF Fusion -> Cross-Encoder -> Context string
                                            |
                                            v
                              [learn_and_test or learn_only]
                                    Explanation Agent (LLaMA 3.3-70B)
                                            |
                                   [learn_and_test or quiz_only]
                                            v
                                      Quiz Agent (LLaMA 3.3-70B)
                                            |
                                            v
                                    Synthesizer merges output
                                            |
                                            v
                              Streamlit renders markdown output
```

---

## Hackathon Context

This project was built for the **HCL Tech Hackathon** under the **Multi-Agent AI Systems** category. The key innovations are:

1. **Hybrid RAG** — combining BM25 sparse retrieval and FAISS dense retrieval with RRF fusion, going beyond simple vector search
2. **Cross-Encoder Reranking** — a two-stage retrieval approach for precision (bi-encoder for recall, cross-encoder for precision)
3. **Lost-in-the-Middle Fix** — research-backed context reordering to maximize LLM attention on the most relevant chunks
4. **Fully Stateful Multi-Agent Orchestration** — using LangGraph's typed StateGraph for clean, inspectable, and extensible agent coordination
5. **Security-First Design** — built-in guardrails against prompt injection and jailbreaking

---

## Team

| Role | Name | Email |
|---|---|---|
| **Team Name** | `PyCoders_____________` | — |
| Participant 1 | `Abhishek Kumar_______` | `abhishek1.kumar22b@iiitg.ac.in_____________________` |
| Participant 2 | `Abhishek Kumar______________________` | `abhishek.kumar22b@iiitg.ac.in______________________` |
| Participant 3 | `Ravi Ranjan______________________` | `ravi.ranjan22b@iiitg.ac.in______________________` |

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 [Team Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) — LLM application framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Multi-agent workflow orchestration
- [Groq](https://groq.com) — Ultra-fast LLM inference API
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Sentence Transformers](https://www.sbert.net/) — HuggingFace embedding models and cross-encoders
- [Streamlit](https://streamlit.io) — Python web UI framework
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation for Python

---

<div align="center">

**Built with passion for the HCL Tech Hackathon 🚀**

*If you found this project useful, please consider giving it a ⭐ star on GitHub!*

</div>

 
 
