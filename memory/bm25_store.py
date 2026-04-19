"""BM25 keyword index over document chunks."""

from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k=10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        return [(self.chunks[i], scores[i]) for i in top_indices]
