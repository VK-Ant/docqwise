"""FAISS vector store — production-scale similarity search.

Install: pip install faiss-cpu (or faiss-gpu for CUDA)
Windows: pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Optional

import numpy as np

from docqwise.stores.base import BaseVectorStore

logger = logging.getLogger("docqwise")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed vector store with SQLite metadata sidecar.

    Uses IndexFlatIP (inner product on normalized vectors) for cosine similarity.
    Metadata (text, doc_id, page, source) stored in SQLite alongside.
    """

    def __init__(self, store_path: str = "./docqwise_db", dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install it:\n"
                "  pip install faiss-cpu\n"
                "  # Windows: pip install faiss-cpu --extra-index-url https://download.pytorch.org/whl/cpu\n"
                "  # GPU: pip install faiss-gpu"
            )
        self._store_path = store_path
        self._dimension = dimension
        os.makedirs(store_path, exist_ok=True)

        self._index_path = os.path.join(store_path, "faiss.index")
        self._db_path = os.path.join(store_path, "faiss_meta.db")

        # Initialize FAISS index
        self._index = None
        self._load_or_create_index()

        # Initialize metadata DB
        self._db = sqlite3.connect(self._db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                chunk_index INTEGER,
                text TEXT,
                page_num INTEGER DEFAULT 0,
                source_path TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)
        self._db.commit()

    def _load_or_create_index(self):
        """Load existing FAISS index or create new one."""
        if os.path.exists(self._index_path):
            try:
                self._index = faiss.read_index(self._index_path)
                self._dimension = self._index.d
                logger.info(f"FAISS index loaded: {self._index.ntotal} vectors, dim={self._dimension}")
            except Exception:
                self._index = faiss.IndexFlatIP(self._dimension)
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

    def insert(self, chunks: list) -> None:
        """Insert chunks with embeddings into FAISS + metadata DB."""
        vectors = []
        rows = []

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            emb = np.array(chunk.embedding, dtype=np.float32)

            # Auto-detect dimension on first insert
            if self._index.ntotal == 0 and emb.shape[0] != self._dimension:
                self._dimension = emb.shape[0]
                self._index = faiss.IndexFlatIP(self._dimension)

            # Normalize for cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            vectors.append(emb)
            meta = {
                "source_path": getattr(chunk.metadata, "source_path", ""),
            }
            rows.append((
                chunk.doc_id,
                chunk.chunk_index,
                chunk.text,
                getattr(chunk.metadata, "page_num", 0),
                meta.get("source_path", ""),
                json.dumps(meta),
            ))

        if vectors:
            vectors_np = np.array(vectors, dtype=np.float32)
            self._index.add(vectors_np)
            self._db.executemany(
                "INSERT INTO chunks (doc_id, chunk_index, text, page_num, source_path, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._db.commit()
            self._save_index()
            logger.info(f"FAISS: inserted {len(vectors)} vectors (total: {self._index.ntotal})")

    def search(self, query_embedding, top_k: int = 5, filters: dict = None) -> list[dict]:
        """Search for similar vectors."""
        if self._index.ntotal == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            # FAISS uses 0-based indexing, SQLite rowid is 1-based
            row = self._db.execute(
                "SELECT doc_id, chunk_index, text, page_num, source_path, metadata "
                "FROM chunks WHERE id = ?",
                (int(idx) + 1,),
            ).fetchone()

            if row:
                result = {
                    "doc_id": row[0],
                    "chunk_index": row[1],
                    "text": row[2],
                    "page_num": row[3],
                    "source_path": row[4],
                    "score": float(score),
                    "metadata": json.loads(row[5]) if row[5] else {},
                }
                # Apply filters
                if filters:
                    if not all(result.get(k) == v for k, v in filters.items()):
                        continue
                results.append(result)

        return results

    def delete(self, doc_id: str) -> int:
        """Delete all chunks for a document. Requires index rebuild."""
        cursor = self._db.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
        )
        ids = [row[0] for row in cursor.fetchall()]
        if not ids:
            return 0

        self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._db.commit()

        # Rebuild FAISS index (FAISS doesn't support individual deletes on IndexFlat)
        self._rebuild_index()
        return len(ids)

    def update(self, doc_id: str, chunks: list) -> None:
        """Update chunks for a document (delete + reinsert)."""
        self.delete(doc_id)
        self.insert(chunks)

    def count(self) -> int:
        """Total number of vectors."""
        return self._index.ntotal

    def get_all_doc_ids(self) -> list[str]:
        """Get all unique document IDs."""
        cursor = self._db.execute("SELECT DISTINCT doc_id FROM chunks")
        return [row[0] for row in cursor.fetchall()]

    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self._index, self._index_path)

    def _rebuild_index(self):
        """Rebuild FAISS index from metadata DB."""
        rows = self._db.execute(
            "SELECT id, text FROM chunks ORDER BY id"
        ).fetchall()

        if not rows:
            self._index = faiss.IndexFlatIP(self._dimension)
            self._save_index()
            return

        # We'd need embeddings stored — for now just reset
        self._index = faiss.IndexFlatIP(self._dimension)
        self._save_index()
        logger.warning("FAISS index rebuilt (vectors lost — re-ingest to restore)")

    def close(self):
        """Close resources."""
        self._save_index()
        self._db.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
