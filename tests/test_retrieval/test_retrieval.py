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

def test_qa_answer_object():
    from docqwise.retrieval.qa_engine import QAAnswer
    a = QAAnswer(answer="100", source="/path/to/invoice.pdf", confidence=0.95, method="llm")
    assert a.source_name == "invoice.pdf"
    assert "Source: invoice.pdf" in str(a)
    d = a.to_dict()
    assert d["answer"] == "100"
    assert d["source_name"] == "invoice.pdf"

def test_qa_multi_source():
    from docqwise.retrieval.qa_engine import QAEngine
    qa = QAEngine()
    qa.register_structured("invoices.csv", ["Vendor", "Total"], [
        {"Vendor": "Acme", "Total": "218300"},
    ])
    qa.register_structured("sales.csv", ["Month", "Total"], [
        {"Month": "Jan", "Total": "5000"},
        {"Month": "Feb", "Total": "7000"},
    ])
    r1 = qa.ask_with_source("What is the total?", source="invoices.csv")
    assert "218,300" in r1.answer
    assert r1.source_name == "invoices.csv"
    r2 = qa.ask_with_source("What is the total?", source="sales.csv")
    assert "12,000" in r2.answer
    assert r2.source_name == "sales.csv"

def test_rag_strategy_modes():
    from docqwise.retrieval.rag_strategy import RAGStrategy
    for mode in ["general", "graphrag", "multimodal"]:
        s = RAGStrategy(mode=mode)
        assert s.mode == mode

def test_rag_strategy_invalid():
    from docqwise.retrieval.rag_strategy import RAGStrategy
    import pytest
    with pytest.raises(ValueError):
        RAGStrategy(mode="invalid")

def test_graphrag_engine():
    from docqwise.retrieval.graphrag import GraphRAGEngine
    engine = GraphRAGEngine()
    graph = engine.get_graph()
    assert graph.node_count == 0

def test_graph_visualizer():
    from docqwise.graph.document_graph import DocumentGraph
    from docqwise.retrieval.graphrag import GraphVisualizer
    g = DocumentGraph()
    g.add_document("inv1", {"type": "invoice"})
    g.add_entity("acme", "ORG", "Acme Corp")
    g.add_edge("inv1", "acme", "from_vendor")
    html = GraphVisualizer.to_html(g)
    assert "vis-network" in html
    assert "Acme Corp" in html
    assert "inv1" in html
