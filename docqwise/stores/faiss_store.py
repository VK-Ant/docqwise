"""FAISS vector store — fast, production-grade, local.

Faster than SQLite for large datasets. Scales to millions of documents.
No server needed. Runs in-process.

Install:
    pip install faiss-cpu            # CPU (works everywhere)
    pip install faiss-gpu            # GPU (NVIDIA CUDA)

Windows fix (if faiss-cpu fails):
    pip install faiss-cpu --no-cache-dir
    # OR
    conda install -c conda-forge faiss-cpu
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from docqwise.core.chunk import DocqwiseChunk
from docqwise.stores.base import BaseVectorStore

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store. Fast cosine similarity search.

    Usage:
        from docqwise.stores.faiss_store import FAISSVectorStore
        store = FAISSVectorStore("./docqwise_db")
        store.insert(chunks)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, store_path: str = "./docqwise_db", dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not installed. Install with:\n"
                "  pip install faiss-cpu\n"
                "  # Windows: pip install faiss-cpu --no-cache-dir\n"
                "  # OR: conda install -c conda-forge faiss-cpu"
            )

        os.makedirs(store_path, exist_ok=True)
        self._store_path = store_path
        self._index_path = os.path.join(store_path, "faiss.index")
        self._meta_path = os.path.join(store_path, "faiss_meta.pkl")
        self._dimension = dimension

        # Metadata storage (chunk_id, doc_id, text, source_path, metadata)
        self._metadata: list[dict] = []

        # Load existing index or create new
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "rb") as f:
                self._metadata = pickle.load(f)
            self._dimension = self._index.d
        else:
            self._index = faiss.IndexFlatIP(dimension)  # inner product (cosine on normalized vectors)

    def _save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

    def insert(self, chunks: list[DocqwiseChunk]) -> None:
        vectors = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue

            emb = chunk.embedding.astype(np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            vectors.append(emb)
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.metadata.doc_id,
                "text": chunk.text,
                "source_path": chunk.metadata.source_path,
                "metadata": {
                    "page_num": chunk.metadata.page_num,
                    "chunk_index": chunk.metadata.chunk_index,
                    "element_type": chunk.metadata.element_type.value,
                    "heading_context": chunk.metadata.heading_context,
                    "confidence": chunk.metadata.confidence,
                },
            })

        if vectors:
            matrix = np.array(vectors, dtype=np.float32)
            # Resize index if dimension changed
            if self._index.d != matrix.shape[1]:
                import faiss
                self._dimension = matrix.shape[1]
                self._index = faiss.IndexFlatIP(self._dimension)
            self._index.add(matrix)
            self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[dict] = None) -> list[dict]:
        if self._index.ntotal == 0:
            return []

        # Normalize query
        query = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        query = query.reshape(1, -1)

        # Search
        scores, indices = self._index.search(query, min(top_k * 3, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]

            # Apply filters
            if filters:
                skip = False
                for k, v in filters.items():
                    if k == "doc_id" and meta.get("doc_id") != v:
                        skip = True
                    elif meta.get("metadata", {}).get(k) != v:
                        skip = True
                if skip:
                    continue

            results.append({
                "chunk_id": meta["chunk_id"],
                "doc_id": meta["doc_id"],
                "text": meta["text"],
                "score": float(score),
                "metadata": meta["metadata"],
                "source_path": meta.get("source_path", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def delete(self, doc_id: str) -> None:
        """Delete all chunks for a document. Rebuilds index."""

        new_meta = []
        new_vectors = []

        for i, meta in enumerate(self._metadata):
            if meta["doc_id"] != doc_id:
                new_meta.append(meta)
                vec = self._index.reconstruct(i)
                new_vectors.append(vec)

        self._metadata = new_meta
        self._index = faiss.IndexFlatIP(self._dimension)
        if new_vectors:
            matrix = np.array(new_vectors, dtype=np.float32)
            self._index.add(matrix)
        self._save()

    def update(self, doc_id: str, chunks: list[DocqwiseChunk]) -> None:
        self.delete(doc_id)
        self.insert(chunks)

    def count(self) -> int:
        return self._index.ntotal

    def get_all_doc_ids(self) -> list[str]:
        return list(set(m["doc_id"] for m in self._metadata))
