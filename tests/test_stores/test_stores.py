"""Tests for vector stores."""
import os
import shutil
import tempfile
import numpy as np
import pytest
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, ElementType

def _make_chunk(chunk_id, doc_id, text, emb):
    return DocqwiseChunk(
        chunk_id=chunk_id, text=text,
        embedding=np.array(emb, dtype=np.float32),
        metadata=ChunkMetadata(doc_id=doc_id, element_type=ElementType.TEXT),
    )

def test_sqlite_store():
    from docqwise.stores.sqlite_store import SQLiteVectorStore
    path = os.path.join(tempfile.gettempdir(), "test_sqlite_store")
    if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
    store = SQLiteVectorStore(path)
    assert store.count() == 0
    store.insert([_make_chunk("c1", "d1", "hello", [0.1, 0.2, 0.3])])
    assert store.count() == 1
    results = store.search(np.array([0.1, 0.2, 0.3], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "hello"
    store.delete("d1")
    assert store.count() == 0
    assert store.get_all_doc_ids() == []
    shutil.rmtree(path, ignore_errors=True)

def test_faiss_store():
    try:
        from docqwise.stores.faiss_store import FAISSVectorStore, FAISS_AVAILABLE
        if not FAISS_AVAILABLE:
            pytest.skip("FAISS not installed")
    except ImportError:
        pytest.skip("FAISS not installed")
    path = os.path.join(tempfile.gettempdir(), "test_faiss_store")
    if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
    store = FAISSVectorStore(path, dimension=3)
    assert store.count() == 0
    store.insert([_make_chunk("c1", "d1", "hello", [0.1, 0.2, 0.3])])
    assert store.count() == 1
    results = store.search(np.array([0.1, 0.2, 0.3], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0]["score"] > 0.9
    store.delete("d1")
    assert store.count() == 0
    shutil.rmtree(path, ignore_errors=True)
