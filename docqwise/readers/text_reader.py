"""Text/Markdown/RST/LOG reader."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class TextReader(BaseReader):
    FORMAT_MAP = {".txt": DocumentType.TEXT, ".md": DocumentType.MARKDOWN,
                  ".rst": DocumentType.TEXT, ".log": DocumentType.TEXT}

    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in self.FORMAT_MAP

    def supported_formats(self) -> list[str]:
        return list(self.FORMAT_MAP.keys())

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        encoding = kwargs.get("encoding", "utf-8")
        with open(path, "r", encoding=encoding, errors="replace") as f:
            text = f.read()
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        ext = os.path.splitext(path)[1].lower()
        doc_type = self.FORMAT_MAP.get(ext, DocumentType.TEXT)
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=doc_type,
            metadata=DocumentMetadata(word_count=len(text.split()), file_size_bytes=file_size,
                                       file_hash=file_hash, has_selectable_text=True),
            text=text, pages=[PageContent(page_num=1, text=text)],
        )
