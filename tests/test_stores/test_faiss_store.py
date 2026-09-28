"""Tests for FAISS vector store."""

import os
import tempfile

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


def test_faiss_store_creation(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    store = FAISSVectorStore(dimension=64)
    assert store is not None


def test_faiss_store_add_and_search(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    store = FAISSVectorStore(dimension=4)
    store.add("doc1", [1.0, 0.0, 0.0, 0.0], {"source": "a.pdf"})
    store.add("doc2", [0.0, 1.0, 0.0, 0.0], {"source": "b.pdf"})

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"


def test_faiss_store_save_load(faiss_available):
    from docqwise.stores.faiss_store import FAISSVectorStore
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_store")

        store = FAISSVectorStore(dimension=4)
        store.add("doc1", [1.0, 0.0, 0.0, 0.0], {"source": "test.pdf"})
        store.save(path)

        store2 = FAISSVectorStore.load(path)
        results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["doc_id"] == "doc1"
