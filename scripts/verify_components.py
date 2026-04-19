"""
Smoke-test each layer. Run from project root:
  python scripts/verify_components.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"), override=True)
    key = os.getenv("GROQ_API_KEY", "")
    groq_ready = bool(key and key != "your_groq_api_key_here" and len(key) > 12)
    print("1) GROQ_API_KEY:", "OK (loaded)" if groq_ready else "MISSING or placeholder — save key in project root `.env`")

    print("2) vector_store (biology.txt) ...")
    from memory.vector_store import (
        build_vector_store,
        faiss_search,
        load_documents_from_file,
        split_documents,
    )

    bio = os.path.join(ROOT, "memory", "sample_notes", "biology.txt")
    docs = load_documents_from_file(bio)
    chunks = split_documents(docs)
    vs = build_vector_store(chunks)
    hits = faiss_search("photosynthesis", vs, k=3)
    assert hits, "FAISS returned no hits"
    print("   OK — FAISS:", len(hits), "chunks")

    print("3) bm25_store ...")
    from memory.bm25_store import BM25Store

    bm = BM25Store(chunks)
    bm_hits = bm.search("chloroplast", k=3)
    assert bm_hits, "BM25 returned no hits"
    print("   OK — BM25:", len(bm_hits), "hits")

    print("4) retriever (cross-encoder; first run may download weights) ...")
    from memory.retriever import retrieve_and_rerank

    ctx = retrieve_and_rerank("What is photosynthesis?", vs, bm)
    assert isinstance(ctx, str) and len(ctx) > 10
    print("   OK — reranked context length", len(ctx))

    print("5) LangGraph import ...")
    from graph.study_graph import study_graph

    assert study_graph is not None
    print("   OK — compiled graph:", type(study_graph).__name__)

    if groq_ready:
        print("6) End-to-end invoke (planner + fast path) ...")
        out = study_graph.invoke(
            {
                "query": "Hi",
                "intent": "",
                "context": "",
                "explanation": "",
                "quiz": "",
                "final_output": "",
                "vector_store": None,
                "bm25_store": None,
            }
        )
        assert out.get("final_output")
        print("   OK —", len(out["final_output"]), "chars")

        print("7) Planner label check ...")
        from agents.planner_agent import classify_intent

        intent = classify_intent("Teach me photosynthesis and test me")
        print("   OK — intent:", intent)
    else:
        print("6–7) Groq live tests SKIPPED — add a real key to `.env` then re-run.")

    print("\nAll runnable checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
