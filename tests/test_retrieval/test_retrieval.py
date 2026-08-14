"""Tests for retrieval components."""

def test_query_router():
    from docqwise.retrieval.query_router import QueryRouter
    r = QueryRouter()
    assert r.route("What is the total amount?") == "computation"
    assert r.route("What is the average?") == "computation"
    assert r.route("How many items?") == "computation"
    assert r.route("Show me overdue invoices") == "filter"
    assert r.route("payment terms and conditions") == "semantic"

def test_bm25():
    from docqwise.retrieval.hybrid_search import BM25
    bm25 = BM25()
    bm25.index(["hello world", "foo bar baz", "hello foo"])
    results = bm25.search("hello", top_k=2)
    assert len(results) == 2
    assert results[0][0] in (0, 2)

def test_qa_engine():
    from docqwise.retrieval.qa_engine import QAEngine
    qa = QAEngine()
    qa.register_structured("test", ["Name", "Amount"], [
        {"Name": "A", "Amount": "100"},
        {"Name": "B", "Amount": "200"},
    ])
    answer = qa.ask("What is the total amount?", source="test")
    assert "300" in answer
