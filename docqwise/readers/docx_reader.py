"""DOCX reader using python-docx."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class DOCXReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return path.lower().endswith((".docx", ".doc"))

    def supported_formats(self) -> list[str]:
        return [".docx", ".doc"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        from docx import Document
        doc = Document(path)
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        words = full_text.split()

        # Extract tables
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                table_texts.append(row_text)

        if table_texts:
            full_text += "\n\n" + "\n".join(table_texts)

        metadata = DocumentMetadata(
            title=doc.core_properties.title or "",
            author=doc.core_properties.author or "",
            page_count=1,
            word_count=len(words),
            file_size_bytes=file_size,
            file_hash=file_hash,
            has_selectable_text=True,
        )

        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path,
            source_type=SourceType.FILE,
            doc_type=DocumentType.DOCX,
            metadata=metadata,
            text=full_text,
            pages=[PageContent(page_num=1, text=full_text)],
        )
