"""PDF reader using PyMuPDF."""

from __future__ import annotations

import hashlib
import os

import fitz  # PyMuPDF

from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType,
    PageContent, SourceType,
)
from docqwise.readers.base import BaseReader


class PDFReader(BaseReader):
    """Read PDF documents using PyMuPDF (fitz)."""

    def can_read(self, path: str) -> bool:
        return path.lower().endswith((".pdf",))

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        """Read a PDF and return DocqwiseDocument with text and metadata."""
        doc = fitz.open(path)
        file_size = os.path.getsize(path)

        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        pages = []
        full_text_parts = []
        total_words = 0
        has_text = False

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                has_text = True

            words = text.split()
            total_words += len(words)
            full_text_parts.append(text)

            pages.append(PageContent(
                page_num=page_num + 1,
                text=text,
                width=page.rect.width,
                height=page.rect.height,
                is_scanned=not bool(text.strip()),
            ))

        metadata = DocumentMetadata(
            title=doc.metadata.get("title", ""),
            author=doc.metadata.get("author", ""),
            page_count=len(doc),
            word_count=total_words,
            file_size_bytes=file_size,
            file_hash=file_hash,
            is_scanned=not has_text,
            has_selectable_text=has_text,
        )

        doc_id = DocqwiseDocument.generate_id(path, file_hash)
        doc.close()

        return DocqwiseDocument(
            doc_id=doc_id,
            source_path=path,
            source_type=SourceType.FILE,
            doc_type=DocumentType.PDF,
            metadata=metadata,
            text="\n".join(full_text_parts),
            pages=pages,
        )
