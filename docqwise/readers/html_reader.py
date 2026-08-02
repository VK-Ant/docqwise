"""HTML reader."""
from __future__ import annotations
import hashlib, os, re
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class HTMLReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".html", ".htm", ".xhtml", ".mhtml")

    def supported_formats(self) -> list[str]:
        return [".html", ".htm", ".xhtml", ".mhtml"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else ""
        except ImportError:
            text = re.sub(r"<[^>]+>", "", content)
            title = ""
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.HTML,
            metadata=DocumentMetadata(title=title, word_count=len(text.split()),
                                       file_size_bytes=file_size, file_hash=file_hash,
                                       has_selectable_text=True),
            text=text, pages=[PageContent(page_num=1, text=text)],
        )
