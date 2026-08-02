"""JSON/JSONL/XML/YAML reader."""
from __future__ import annotations
import hashlib, json, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class JSONReader(BaseReader):
    FORMAT_MAP = {".json": DocumentType.JSON, ".jsonl": DocumentType.JSON,
                  ".xml": DocumentType.XML, ".yaml": DocumentType.YAML,
                  ".yml": DocumentType.YAML, ".toml": DocumentType.YAML}

    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in self.FORMAT_MAP

    def supported_formats(self) -> list[str]:
        return list(self.FORMAT_MAP.keys())

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        file_size = os.path.getsize(path)
        with open(path, "rb") as fb:
            file_hash = hashlib.sha256(fb.read()).hexdigest()[:16]
        ext = os.path.splitext(path)[1].lower()
        doc_type = self.FORMAT_MAP.get(ext, DocumentType.JSON)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = None
        if ext in (".json",):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                pass
        elif ext == ".jsonl":
            parsed = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                parsed = yaml.safe_load(content)
            except Exception:
                pass
        elif ext == ".xml":
            pass  # keep as text
        text = json.dumps(parsed, indent=2, default=str) if parsed else content
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=doc_type,
            metadata=DocumentMetadata(word_count=len(text.split()), file_size_bytes=file_size,
                                       file_hash=file_hash, has_selectable_text=True,
                                       custom={"parsed": parsed}),
            text=text, pages=[PageContent(page_num=1, text=text)],
        )
