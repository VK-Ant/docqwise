"""Core document representation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    EXCEL = "excel"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    HTML = "html"
    PPTX = "pptx"
    EMAIL = "email"
    TEXT = "text"
    PARQUET = "parquet"
    DATABASE = "database"
    RTF = "rtf"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    FILE = "file"
    URL = "url"
    DATABASE = "database"
    CLOUD_STORAGE = "cloud_storage"
    EMAIL = "email"
    STREAM = "stream"


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""

    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    page_count: int = 0
    word_count: int = 0
    language: Optional[str] = None
    file_size_bytes: int = 0
    file_hash: str = ""
    is_scanned: bool = False
    has_selectable_text: bool = True
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageContent:
    """Content of a single page."""

    page_num: int
    text: str = ""
    width: float = 0.0
    height: float = 0.0
    elements: list[Any] = field(default_factory=list)
    is_scanned: bool = False


@dataclass
class DocqwiseDocument:
    """Core document representation output of reader and extractors."""

    doc_id: str
    source_path: str
    source_type: SourceType
    doc_type: DocumentType
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    # Extracted content
    text: str = ""
    pages: list[PageContent] = field(default_factory=list)
    tables: list[Any] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    entities: list[Any] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)
    form_fields: dict[str, Any] = field(default_factory=dict)

    # Processing state
    extraction_strategy: str = ""
    extraction_confidence: float = 0.0
    processing_time_ms: float = 0.0
    extracted_at: Optional[datetime] = None

    @staticmethod
    def generate_id(source_path: str, content_hash: str = "") -> str:
        """Generate a unique document ID."""
        raw = f"{source_path}:{content_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
