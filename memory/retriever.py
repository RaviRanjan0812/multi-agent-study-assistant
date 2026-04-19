"""RRF fusion (BM25 + FAISS), cross-encoder reranking, context reorder."""

from sentence_transformers import CrossEncoder

from memory.vector_store import faiss_search

# Lightest cross-encoder — fast on CPU, good enough for hackathon
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")


def rrf_fusion(faiss_docs, bm25_results, k=60):
    """
    Reciprocal Rank Fusion.
    Combines FAISS and BM25 ranked lists without hand-tuned blend weights.
    RRF score = Σ 1/(k + rank_i) for each list the doc appears in.
    Standard k=60.
    """
    scores = {}

    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_results):
        key = doc.page_content
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    # Build unified doc map (dedup by content)
    all_docs = {doc.page_content: doc for doc in faiss_docs}
    for doc, _ in bm25_results:
        all_docs[doc.page_content] = doc

    ranked = sorted(
        all_docs.values(),
        key=lambda d: scores.get(d.page_content, 0),
        reverse=True,
    )
    return ranked[:10]  # top 10 fused candidates


def reorder_context(docs):
    """
    Lost-in-the-Middle fix.
    LLMs best attend to content at the START and END of context.
    Put best chunk first, second-best last, rest in the middle.
    """
    if len(docs) <= 2:
        return docs
    best = docs[0]
    second_best = docs[1]
    middle = docs[2:]
    return [best] + middle + [second_best]


def cross_encoder_rerank(query: str, docs, top_k=3):
    """
    Precisely scores each (query, doc) pair.
    Returns top_k chunks as a single string ready for the LLM.
    """
    if not docs:
        return "No relevant content found. Use general knowledge."

    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:top_k]]
    reordered = reorder_context(top_docs)
    return "\n\n".join(d.page_content for d in reordered)


def retrieve_and_rerank(query: str, vector_store, bm25_store) -> str:
    """
    Full pipeline:
      1. BM25 keyword search → top 10
      2. FAISS semantic search → top 10
      3. RRF fusion → unified top 10
      4. Cross-Encoder rerank → top 3
      5. Context reorder (Lost-in-the-Middle fix)
    Returns a single string context for the LLM.
    """
    if vector_store is None or bm25_store is None:
        return "No document uploaded. Use general knowledge."

    faiss_docs = faiss_search(query, vector_store, k=10)
    bm25_results = bm25_store.search(query, k=10)

    if not faiss_docs and not bm25_results:
        return "No specific notes found. Use general knowledge."

    fused = rrf_fusion(faiss_docs, bm25_results)
    return cross_encoder_rerank(query, fused, top_k=3)


# Backward-compatible name for callers expecting `retrieve`
retrieve = retrieve_and_rerank
