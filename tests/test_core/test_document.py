"""Tests for core document models."""

from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, SourceType,
)
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, BoundingBox, ElementType
from docqwise.core.field import FieldResult, ExtractionResult
from docqwise.core.element import Table, TableCell, Entity, Relation, FormField


def test_document_creation():
    doc = DocqwiseDocument(
        doc_id="test123",
        source_path="test.pdf",
        source_type=SourceType.FILE,
        doc_type=DocumentType.PDF,
    )
    assert doc.doc_id == "test123"
    assert doc.doc_type == DocumentType.PDF
    assert doc.text == ""
    assert doc.tables == []


def test_document_generate_id():
    doc_id = DocqwiseDocument.generate_id("test.pdf", "abc123")
    assert isinstance(doc_id, str)
    assert len(doc_id) == 16
    # Same input = same ID
    assert doc_id == DocqwiseDocument.generate_id("test.pdf", "abc123")


def test_bounding_box():
    bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80, page=1)
    assert bbox.width == 90
    assert bbox.height == 60
    assert bbox.area == 5400
    assert bbox.to_list() == [10, 20, 100, 80]


def test_chunk_creation():
    chunk = DocqwiseChunk(
        chunk_id="chunk_001",
        text="This is a test chunk",
        metadata=ChunkMetadata(
            doc_id="test123",
            page_num=1,
            element_type=ElementType.TEXT,
        ),
    )
    assert chunk.chunk_id == "chunk_001"
    assert chunk.metadata.element_type == ElementType.TEXT


def test_field_result():
    field = FieldResult(
        value="Acme Corp",
        confidence=0.97,
        raw_text="Acme Corp.",
        page=1,
    )
    assert field.value == "Acme Corp"
    assert field.confidence == 0.97
    assert not field.is_corrected


def test_extraction_result_to_dict():
    result = ExtractionResult(
        doc_id="test123",
        source_path="invoice.pdf",
        fields={
            "vendor": FieldResult(value="Acme", confidence=0.95),
            "total": FieldResult(value=1500.0, confidence=0.99),
        },
        confidence=0.97,
    )
    d = result.to_dict()
    assert d["vendor"] == "Acme"
    assert d["total"] == 1500.0


def test_extraction_result_correct():
    result = ExtractionResult(
        doc_id="test123",
        source_path="invoice.pdf",
        fields={
            "vendor": FieldResult(value="Acme", confidence=0.95),
            "total": FieldResult(value=1500.0, confidence=0.99),
        },
        confidence=0.97,
    )
    result.correct({"total": 1600.0})
    assert result.fields["total"].value == 1600.0
    assert result.fields["total"].is_corrected is True
    assert result.fields["total"].confidence == 1.0


def test_table_cell():
    cell = TableCell(text="$1,500", row=0, col=1, is_header=False)
    assert cell.text == "$1,500"
    assert cell.row_span == 1


def test_entity():
    entity = Entity(text="Acme Corp", entity_type="ORG", confidence=0.98)
    assert entity.entity_type == "ORG"


def test_relation():
    rel = Relation(subject="Acme Corp", predicate="obligated_to", object="deliver by March 15")
    assert rel.predicate == "obligated_to"


def test_form_field():
    ff = FormField(field_name="agree_terms", field_type="checkbox", is_checked=True)
    assert ff.is_checked is True


def test_document_types():
    assert DocumentType.PDF.value == "pdf"
    assert DocumentType.EXCEL.value == "excel"
    assert DocumentType.DATABASE.value == "database"


def test_metadata_defaults():
    meta = DocumentMetadata()
    assert meta.page_count == 0
    assert meta.is_scanned is False
    assert meta.has_selectable_text is True
