"""Core data models for docqwise."""

from docqwise.core.document import DocqwiseDocument, DocumentMetadata, DocumentType, SourceType
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, ElementType, BoundingBox
from docqwise.core.field import FieldResult, ExtractionResult
from docqwise.core.element import Table, TableCell, ExtractedImage, Entity, Relation, FormField

__all__ = [
    "DocqwiseDocument", "DocumentMetadata", "DocumentType", "SourceType",
    "DocqwiseChunk", "ChunkMetadata", "ElementType", "BoundingBox",
    "FieldResult", "ExtractionResult",
    "Table", "TableCell", "ExtractedImage", "Entity", "Relation", "FormField",
]
