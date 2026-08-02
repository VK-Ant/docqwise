"""Main retrieval engine."""
from __future__ import annotations
from typing import Optional
import numpy as np

class Retriever:
    def __init__(self, vector_store, embedder):
        self._store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[dict]:
        query_emb = self._embedder.embed(query)
        return self._store.search(query_emb, top_k=top_k, filters=filters)
