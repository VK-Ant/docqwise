"""Tests for RAG strategy and QAAnswer."""

from docqwise.retrieval.rag_strategy import QAAnswer


def test_qa_answer_creation():
    answer = QAAnswer(
        answer="The total is $1,500",
        source="invoice.pdf",
        confidence=0.95,
        method="graphrag",
    )
    assert answer.answer == "The total is $1,500"
    assert answer.confidence == 0.95
    assert answer.method == "graphrag"


def test_qa_answer_source_name():
    answer = QAAnswer(
        answer="test",
        source="path/to/invoice.pdf",
        confidence=0.9,
        method="general",
    )
    assert answer.source_name == "invoice.pdf"


def test_qa_answer_no_source():
    answer = QAAnswer(
        answer="test",
        source=None,
        confidence=0.5,
        method="general",
    )
    assert answer.source_name == ""


def test_qa_answer_with_evidence():
    evidence = [
        {"entity": "$15,750.00", "type": "MONEY", "sources": ["inv.pdf"]},
        {"entity": "Acme Corp", "type": "ORG", "sources": ["inv.pdf"]},
    ]
    answer = QAAnswer(
        answer="Total is $15,750",
        source="inv.pdf",
        confidence=0.92,
        method="graphrag",
        evidence=evidence,
    )
    assert len(answer.evidence) == 2
    assert answer.evidence[0]["type"] == "MONEY"


def test_qa_answer_repr():
    answer = QAAnswer(
        answer="The total is $1,500",
        source="invoice.pdf",
        confidence=0.95,
        method="general",
    )
    r = repr(answer)
    assert "QAAnswer" in r
    assert "0.95" in r
