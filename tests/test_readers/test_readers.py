"""Tests for all document readers."""
import os
import pytest

DEMO = "demo"

def test_pdf_reader():
    from docqwise.readers.pdf_reader import PDFReader
    r = PDFReader()
    assert r.can_read("test.pdf")
    assert not r.can_read("test.docx")
    doc = r.read(os.path.join(DEMO, "sample_invoice.pdf"))
    assert doc.metadata.page_count == 1
    assert doc.metadata.word_count > 10
    assert len(doc.text) > 50
    assert len(doc.metadata.file_hash) == 16
    assert doc.doc_type.value == "pdf"

def test_csv_reader():
    from docqwise.readers.csv_reader import CSVReader
    r = CSVReader()
    doc = r.read(os.path.join(DEMO, "sample_sales.csv"))
    assert doc.metadata.custom["row_count"] == 10
    assert len(doc.metadata.custom["headers"]) > 0
    assert doc.doc_type.value == "csv"

def test_json_reader():
    from docqwise.readers.json_reader import JSONReader
    r = JSONReader()
    doc = r.read(os.path.join(DEMO, "sample_data.json"))
    assert len(doc.text) > 0
    assert doc.metadata.custom.get("parsed") is not None
    assert doc.doc_type.value == "json"

def test_auto_reader_detection():
    from docqwise.readers.auto_reader import AutoReader
    from docqwise.core.document import DocumentType, SourceType
    auto = AutoReader()
    assert auto.detect_type("doc.pdf") == DocumentType.PDF
    assert auto.detect_type("doc.docx") == DocumentType.DOCX
    assert auto.detect_type("img.jpg") == DocumentType.IMAGE
    assert auto.detect_type("data.csv") == DocumentType.CSV
    assert auto.detect_type("data.json") == DocumentType.JSON
    assert auto.detect_type("data.xlsx") == DocumentType.EXCEL
    assert auto.detect_type("page.html") == DocumentType.HTML
    assert auto.detect_type("slides.pptx") == DocumentType.PPTX
    assert auto.detect_type("mail.eml") == DocumentType.EMAIL
    assert auto.detect_type("data.parquet") == DocumentType.PARQUET
    assert auto.detect_type("postgresql://h/db") == DocumentType.DATABASE
    assert auto.detect_source_type("file.pdf") == SourceType.FILE
    assert auto.detect_source_type("https://x.com") == SourceType.URL
    assert auto.detect_source_type("s3://bucket/") == SourceType.CLOUD_STORAGE

def test_auto_reader_formats():
    from docqwise.readers.auto_reader import AutoReader
    auto = AutoReader()
    assert len(auto.supported_formats()) >= 20
