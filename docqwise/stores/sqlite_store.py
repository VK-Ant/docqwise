"""SQLite + numpy vector store (default, zero external dependencies)."""
from __future__ import annotations
import json, os, sqlite3
from typing import Optional
import numpy as np
from docqwise.stores.base import BaseVectorStore
from docqwise.core.chunk import DocqwiseChunk

class SQLiteVectorStore(BaseVectorStore):
    def __init__(self, store_path: str = "./docqwise_db"):
        os.makedirs(store_path, exist_ok=True)
        self._db_path = os.path.join(store_path, "vectors.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                source_path TEXT
            )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
            conn.commit()
        finally:
            conn.close()

    def insert(self, chunks: list[DocqwiseChunk]) -> None:
        conn = self._get_conn()
        try:
            for chunk in chunks:
                emb_bytes = chunk.embedding.tobytes() if chunk.embedding is not None else None
                meta = json.dumps({
                    "page_num": chunk.metadata.page_num,
                    "chunk_index": chunk.metadata.chunk_index,
                    "element_type": chunk.metadata.element_type.value,
                    "heading_context": chunk.metadata.heading_context,
                    "confidence": chunk.metadata.confidence,
                })
                conn.execute(
                    "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, text, embedding, metadata, source_path) VALUES (?,?,?,?,?,?)",
                    (chunk.chunk_id, chunk.metadata.doc_id, chunk.text, emb_bytes, meta, chunk.metadata.source_path),
                )
            conn.commit()
        finally:
            conn.close()

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[dict] = None) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT chunk_id, doc_id, text, embedding, metadata, source_path FROM chunks").fetchall()
        finally:
            conn.close()
        if not rows:
            return []
        results = []
        for row in rows:
            chunk_id, doc_id, text, emb_bytes, meta_str, source_path = row
            if emb_bytes is None:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape != query_embedding.shape:
                continue
            similarity = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-10))
            meta = json.loads(meta_str) if meta_str else {}
            if filters:
                skip = False
                for k, v in filters.items():
                    if meta.get(k) != v and (k == "doc_id" and doc_id != v):
                        skip = True
                        break
                if skip:
                    continue
            results.append({"chunk_id": chunk_id, "doc_id": doc_id, "text": text,
                            "score": similarity, "metadata": meta, "source_path": source_path})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, doc_id: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.commit()
        finally:
            conn.close()

    def update(self, doc_id: str, chunks: list[DocqwiseChunk]) -> None:
        self.delete(doc_id)
        self.insert(chunks)

    def count(self) -> int:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()

    def get_all_doc_ids(self) -> list[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT DISTINCT doc_id FROM chunks").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
