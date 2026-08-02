"""Image reader (JPG, PNG, TIFF, BMP, WebP)."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class ImageReader(BaseReader):
    FORMATS = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".heic"]

    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in self.FORMATS

    def supported_formats(self) -> list[str]:
        return self.FORMATS

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        from PIL import Image
        img = Image.open(path)
        width, height = img.size
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.IMAGE,
            metadata=DocumentMetadata(page_count=1, file_size_bytes=file_size,
                                       file_hash=file_hash, is_scanned=True, has_selectable_text=False),
            text="", pages=[PageContent(page_num=1, text="", width=width, height=height, is_scanned=True)],
        )
