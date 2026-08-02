"""PPTX reader."""
from __future__ import annotations
import hashlib, os
from docqwise.readers.base import BaseReader
from docqwise.core.document import (
    DocqwiseDocument, DocumentMetadata, DocumentType, PageContent, SourceType,
)

class PPTXReader(BaseReader):
    def can_read(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in (".pptx", ".ppt")

    def supported_formats(self) -> list[str]:
        return [".pptx", ".ppt"]

    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        from pptx import Presentation
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        prs = Presentation(path)
        pages, all_text = [], []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text.strip():
                            slide_text.append(para.text)
            text = "\n".join(slide_text)
            pages.append(PageContent(page_num=i + 1, text=text))
            all_text.append(text)
        full_text = "\n\n".join(all_text)
        return DocqwiseDocument(
            doc_id=DocqwiseDocument.generate_id(path, file_hash),
            source_path=path, source_type=SourceType.FILE, doc_type=DocumentType.PPTX,
            metadata=DocumentMetadata(page_count=len(prs.slides), word_count=len(full_text.split()),
                                       file_size_bytes=file_size, file_hash=file_hash, has_selectable_text=True),
            text=full_text, pages=pages,
        )
