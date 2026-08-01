"""Auto-detect document format and dispatch to appropriate reader."""

from __future__ import annotations

import os
from typing import Optional

from docqwise.core.document import DocqwiseDocument, DocumentType, SourceType
from docqwise.readers.base import BaseReader


FORMAT_MAP = {
    ".pdf": DocumentType.PDF,
    ".doc": DocumentType.DOCX,
    ".docx": DocumentType.DOCX,
    ".rtf": DocumentType.RTF,
    ".txt": DocumentType.TEXT,
    ".md": DocumentType.MARKDOWN,
    ".rst": DocumentType.TEXT,
    ".log": DocumentType.TEXT,
    ".html": DocumentType.HTML,
    ".htm": DocumentType.HTML,
    ".xhtml": DocumentType.HTML,
    ".jpg": DocumentType.IMAGE,
    ".jpeg": DocumentType.IMAGE,
    ".png": DocumentType.IMAGE,
    ".tiff": DocumentType.IMAGE,
    ".tif": DocumentType.IMAGE,
    ".bmp": DocumentType.IMAGE,
    ".webp": DocumentType.IMAGE,
    ".heic": DocumentType.IMAGE,
    ".xlsx": DocumentType.EXCEL,
    ".xls": DocumentType.EXCEL,
    ".xlsm": DocumentType.EXCEL,
    ".ods": DocumentType.EXCEL,
    ".csv": DocumentType.CSV,
    ".tsv": DocumentType.CSV,
    ".json": DocumentType.JSON,
    ".jsonl": DocumentType.JSON,
    ".xml": DocumentType.XML,
    ".yaml": DocumentType.YAML,
    ".yml": DocumentType.YAML,
    ".toml": DocumentType.YAML,
    ".parquet": DocumentType.PARQUET,
    ".pptx": DocumentType.PPTX,
    ".ppt": DocumentType.PPTX,
    ".eml": DocumentType.EMAIL,
    ".msg": DocumentType.EMAIL,
}


class AutoReader(BaseReader):
    """Auto-detects document format and dispatches to the correct reader."""

    def __init__(self):
        self._readers: dict[DocumentType, BaseReader] = {}

    def can_read(self, path: str) -> bool:
        return True

    def supported_formats(self) -> list[str]:
        return list(FORMAT_MAP.keys())

    def detect_type(self, path: str) -> DocumentType:
        """Detect document type from file extension and magic bytes."""
        if path.startswith(("postgresql://", "mysql://", "sqlite:///",
                           "mongodb://", "mongodb+srv://")):
            return DocumentType.DATABASE

        ext = os.path.splitext(path)[1].lower()
        return FORMAT_MAP.get(ext, DocumentType.UNKNOWN)

    def detect_source_type(self, path: str) -> SourceType:
        """Detect source type from path."""
        if path.startswith(("http://", "https://")):
            return SourceType.URL
        if path.startswith(("postgresql://", "mysql://", "sqlite:///", "mongodb://")):
            return SourceType.DATABASE
        if path.startswith(("s3://", "gs://", "az://")):
            return SourceType.CLOUD_STORAGE
        if path.startswith(("imap://", "pop3://")):
            return SourceType.EMAIL
        return SourceType.FILE

    def _get_reader(self, doc_type: DocumentType) -> BaseReader:
        """Get or lazy-load the reader for a document type."""
        if doc_type in self._readers:
            return self._readers[doc_type]

        reader: Optional[BaseReader] = None

        if doc_type == DocumentType.PDF:
            from docqwise.readers.pdf_reader import PDFReader
            reader = PDFReader()
        elif doc_type == DocumentType.DOCX:
            from docqwise.readers.docx_reader import DOCXReader
            reader = DOCXReader()
        elif doc_type == DocumentType.TEXT or doc_type == DocumentType.MARKDOWN:
            from docqwise.readers.text_reader import TextReader
            reader = TextReader()
        elif doc_type == DocumentType.IMAGE:
            from docqwise.readers.image_reader import ImageReader
            reader = ImageReader()
        elif doc_type in (DocumentType.CSV, DocumentType.EXCEL):
            from docqwise.readers.csv_reader import CSVReader
            reader = CSVReader()
        elif doc_type in (DocumentType.JSON, DocumentType.XML, DocumentType.YAML):
            from docqwise.readers.json_reader import JSONReader
            reader = JSONReader()

        if reader:
            self._readers[doc_type] = reader

        return reader

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        """Read a document by auto-detecting its format."""
        doc_type = self.detect_type(path)
        reader = self._get_reader(doc_type)

        if reader is None:
            from docqwise.exceptions import UnsupportedFormatError
            raise UnsupportedFormatError(
                f"No reader available for {doc_type.value} format: {path}"
            )

        return reader.read(path, **kwargs)
