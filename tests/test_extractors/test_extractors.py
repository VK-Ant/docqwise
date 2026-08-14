"""Tests for extractors."""
import os
import pytest

DEMO = "demo"

def test_regex_field_extractor():
    from docqwise.readers.pdf_reader import PDFReader
    from docqwise.extractors.field_extractor import RegexFieldExtractor
    doc = PDFReader().read(os.path.join(DEMO, "sample_invoice.pdf"))
    extractor = RegexFieldExtractor()
    result = extractor.extract_fields(doc)
    assert len(result.fields) > 0
    assert result.strategy_used == "regex"

def test_regex_with_schema():
    from docqwise.readers.pdf_reader import PDFReader
    from docqwise.extractors.field_extractor import RegexFieldExtractor
    doc = PDFReader().read(os.path.join(DEMO, "sample_invoice.pdf"))
    schema = {"invoice_number": {"type": "string"}, "date": {"type": "date"}}
    result = RegexFieldExtractor().extract_fields(doc, schema=schema)
    assert len(result.fields) > 0

def test_entity_extractor():
    from docqwise.readers.pdf_reader import PDFReader
    from docqwise.extractors.entity_extractor import RegexEntityExtractor
    doc = PDFReader().read(os.path.join(DEMO, "sample_invoice.pdf"))
    entities = RegexEntityExtractor().extract_entities(doc)
    assert len(entities) > 0
    assert any(e.entity_type == "DATE" for e in entities)

def test_table_extractor():
    from docqwise.readers.pdf_reader import PDFReader
    from docqwise.extractors.table_extractor import RuleBasedTableExtractor
    doc = PDFReader().read(os.path.join(DEMO, "sample_invoice.pdf"))
    tables = RuleBasedTableExtractor().extract_tables(doc)
    assert isinstance(tables, list)

def test_metadata_extractor():
    from docqwise.readers.pdf_reader import PDFReader
    from docqwise.extractors.metadata_extractor import MetadataExtractor
    doc = PDFReader().read(os.path.join(DEMO, "sample_invoice.pdf"))
    meta = MetadataExtractor().extract(doc)
    assert meta["doc_type"] == "pdf"
    assert meta["page_count"] == 1
