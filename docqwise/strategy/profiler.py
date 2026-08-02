"""Quick document profiler."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument

class DocumentProfiler:
    def profile(self, document: DocqwiseDocument) -> dict:
        text = document.text
        return {
            "doc_type": document.doc_type.value,
            "page_count": document.metadata.page_count,
            "word_count": document.metadata.word_count,
            "is_scanned": document.metadata.is_scanned,
            "has_text": bool(text.strip()),
            "has_tables": "|" in text and text.count("|") > 4,
            "text_density": document.metadata.word_count / max(document.metadata.page_count, 1),
            "language": document.metadata.language,
        }
