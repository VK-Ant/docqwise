"""Text extraction from documents."""
from __future__ import annotations
from docqwise.extractors.base import BaseExtractor
from docqwise.core.document import DocqwiseDocument

class TextExtractor(BaseExtractor):
    def extract(self, document: DocqwiseDocument, **kwargs) -> DocqwiseDocument:
        remove_headers = kwargs.get("remove_headers", False)
        remove_footers = kwargs.get("remove_footers", False)
        text = document.text
        if remove_headers or remove_footers:
            lines = text.split("\n")
            if remove_headers and len(lines) > 2:
                lines = lines[1:]
            if remove_footers and len(lines) > 2:
                lines = lines[:-1]
            text = "\n".join(lines)
        document.text = text
        return document
