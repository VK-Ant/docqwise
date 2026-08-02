"""Parquet/Arrow reader."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class ParquetReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".parquet", ".arrow", ".feather")

    def supported_formats(self) -> list[str]:
        return [".parquet", ".arrow", ".feather"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        import pyarrow.parquet as pq
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        table = pq.read_table(path)
        df = table.to_pandas()
        headers = list(df.columns)
        text_parts = [" | ".join(headers)]
        rows = df.to_dict(orient="records")
        for row in rows:
            text_parts.append(" | ".join(str(v) for v in row.values()))
        full_text = "\n".join(text_parts)
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.PARQUET,
            metadata=DocumentMetadata(word_count=len(full_text.split()), file_size_bytes=file_size,
                                       file_hash=file_hash, has_selectable_text=True,
                                       custom={"headers": headers, "row_count": len(rows), "rows": rows}),
            text=full_text, pages=[PageContent(page_num=1, text=full_text)],
        )
