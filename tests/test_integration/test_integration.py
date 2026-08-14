"""Integration tests — full engine end-to-end."""
import os
import shutil
import tempfile
import pytest
from docqwise import Docqwise

@pytest.fixture
def dq():
    path = os.path.join(tempfile.gettempdir(), "test_integration")
    if os.path.exists(path): shutil.rmtree(path, ignore_errors=True)
    engine = Docqwise(store_path=path)
    yield engine
    shutil.rmtree(path, ignore_errors=True)

def test_ingest_pdf(dq):
    r = dq.ingest("demo/sample_invoice.pdf", embed=False)
    assert r["processed"] == 1

def test_ingest_csv(dq):
    r = dq.ingest("demo/sample_sales.csv", embed=False)
    assert r["processed"] == 1

def test_incremental_skip(dq):
    dq.ingest("demo/sample_invoice.pdf", embed=False)
    r = dq.ingest("demo/sample_invoice.pdf", embed=False)
    assert r["processed"] == 0

def test_extract_fields_regex(dq):
    r = dq.extract_fields("demo/sample_invoice.pdf", template="invoice", method="regex")
    assert len(r.fields) > 0

def test_extract_text(dq):
    text = dq.extract_text("demo/sample_invoice.pdf")
    assert len(text) > 50

def test_classify(dq):
    labels = dq.classify("demo/sample_invoice.pdf")
    assert labels[0]["label"] == "invoice"

def test_compare(dq):
    diff = dq.compare("demo/sample_invoice.pdf", "demo/sample_contract.pdf")
    assert "similarity" in diff

def test_detect_schema(dq):
    s = dq.detect_schema("demo/sample_sales.csv")
    assert len(s["fields"]) > 0

def test_ask(dq):
    dq.ingest("demo/sample_sales.csv", embed=False)
    answer = dq.ask("What is the total amount?", source="demo/sample_sales.csv")
    assert len(answer) > 0

def test_detect_pii(dq):
    pii = dq.detect_pii("demo/sample_invoice.pdf")
    assert isinstance(pii, list)

def test_metrics(dq):
    m = dq.metrics()
    assert m["version"] == "0.2.0"

def test_learning(dq):
    r = dq.extract_fields("demo/sample_invoice.pdf", template="invoice", method="regex")
    r.correct({"tax": 33300.0})
    report = dq.learning_report()
    assert report["total_corrections"] > 0

def test_factory_methods():
    from docqwise.factory import ExtractorFactory, TemplateFactory, StoreFactory
    assert "rag" in ExtractorFactory.available_methods()
    assert "invoice" in TemplateFactory.available()
    assert "faiss" in StoreFactory.STORES
