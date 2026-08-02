"""CSV/TSV reader."""
from __future__ import annotations
import csv, hashlib, os, io
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class CSVReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".csv", ".tsv", ".psv")

    def supported_formats(self) -> list[str]:
        return [".csv", ".tsv", ".psv"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        ext = os.path.splitext(path)[1].lower()
        delimiter = {"csv": ",", ".tsv": "\t", ".psv": "|"}.get(ext, ",")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)
        headers = reader.fieldnames or []
        text_parts = [" | ".join(headers)]
        for row in rows:
            text_parts.append(" | ".join(str(row.get(h, "")) for h in headers))
        full_text = "\n".join(text_parts)
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.CSV,
            metadata=DocumentMetadata(word_count=len(full_text.split()), file_size_bytes=file_size,
                                       file_hash=file_hash, has_selectable_text=True,
                                       custom={"headers": headers, "row_count": len(rows), "rows": rows}),
            text=full_text, pages=[PageContent(page_num=1, text=full_text)],
        )
