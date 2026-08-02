"""Hybrid search combining vector + BM25 keyword search."""
from __future__ import annotations
import math, re
from collections import Counter
from typing import Optional

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs = []
        self._doc_freqs = Counter()
        self._avg_dl = 0.0

    def index(self, documents: list[str]):
        self._docs = [self._tokenize(d) for d in documents]
        self._avg_dl = sum(len(d) for d in self._docs) / max(len(self._docs), 1)
        for doc in self._docs:
            seen = set()
            for token in doc:
                if token not in seen:
                    self._doc_freqs[token] += 1
                    seen.add(token)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_tokens = self._tokenize(query)
        n = len(self._docs)
        scores = []
        for i, doc in enumerate(self._docs):
            score = 0.0
            doc_len = len(doc)
            freq = Counter(doc)
            for qt in q_tokens:
                if qt not in freq:
                    continue
                tf = freq[qt]
                df = self._doc_freqs.get(qt, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avg_dl, 1)))
                score += idf * tf_norm
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())

class HybridSearcher:
    def __init__(self, vector_store, embedder, vector_weight: float = 0.7):
        self._store = vector_store
        self._embedder = embedder
        self._bm25 = BM25()
        self._vector_weight = vector_weight
        self._keyword_weight = 1 - vector_weight
        self._indexed_texts = []
        self._indexed_meta = []

    def build_index(self, texts: list[str], metadata: list[dict] = None):
        self._indexed_texts = texts
        self._indexed_meta = metadata or [{} for _ in texts]
        self._bm25.index(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vector_results = self._store.search(self._embedder.embed(query), top_k=top_k * 2)
        keyword_results = self._bm25.search(query, top_k=top_k * 2)
        combined = {}
        max_vec = max((r["score"] for r in vector_results), default=1.0) or 1.0
        for r in vector_results:
            combined[r["chunk_id"]] = {**r, "final_score": (r["score"] / max_vec) * self._vector_weight}
        max_kw = max((s for _, s in keyword_results), default=1.0) or 1.0
        for idx, score in keyword_results:
            if idx < len(self._indexed_texts):
                text = self._indexed_texts[idx]
                for cid, r in combined.items():
                    if r["text"] == text:
                        r["final_score"] += (score / max_kw) * self._keyword_weight
                        break
        results = sorted(combined.values(), key=lambda x: x.get("final_score", 0), reverse=True)
        return results[:top_k]
