"""Tests for FAISS vector store."""

import os
import tempfile

import numpy as np
import pytest


def test_faiss_import():
    """Test that FAISS store can be imported."""
    from docqwise.stores.faiss_store import FAISSVectorStore
    assert FAISSVectorStore is not None


@pytest.fixture
def faiss_available():
    try:
        import faiss
        return True
    except ImportError:
        pytest.skip("faiss-cpu not installed")


def _make_chunk(doc_id, chunk_index, text, embedding):
    """Create a DocqwiseChunk for testing."""
    from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata
    return DocqwiseChunk(
        chunk_id=f"{doc_id}_{chunk_index}",
        text=text,
        embedding=np.array(embedding, dtype=np.float32),
        metadata=ChunkMetadata(
            doc_id=doc_id,
            source_path=f"{doc_id}.pdf",
            chunk_index=chunk_index,
        ),
    )


def test_faiss_store_creation(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    tmpdir = tempfile.mkdtemp()
    store = FAISSVectorStore(store_path=tmpdir, dimension=64)
    assert store is not None
    store.close()


def test_faiss_store_insert_and_search(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    tmpdir = tempfile.mkdtemp()
    store = FAISSVectorStore(store_path=tmpdir, dimension=4)

    chunks = [
        _make_chunk("doc1", 0, "first document", [1.0, 0.0, 0.0, 0.0]),
        _make_chunk("doc2", 0, "second document", [0.0, 1.0, 0.0, 0.0]),
    ]
    store.insert(chunks)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    store.close()


def test_faiss_store_persistence(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    tmpdir = tempfile.mkdtemp()

    store = FAISSVectorStore(store_path=tmpdir, dimension=4)
    chunks = [_make_chunk("doc1", 0, "test content", [1.0, 0.0, 0.0, 0.0])]
    store.insert(chunks)
    store.close()

    # Re-open from same path
    store2 = FAISSVectorStore(store_path=tmpdir, dimension=4)
    results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    store2.close()
