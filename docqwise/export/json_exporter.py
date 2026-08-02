"""JSON exporter."""
from __future__ import annotations
import json, os
from docqwise.core.document import DocqwiseDocument

class JSONExporter:
    def export(self, documents: list[DocqwiseDocument], path: str, **kwargs):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        for doc in documents:
            data.append({
                "doc_id": doc.doc_id, "source_path": doc.source_path,
                "doc_type": doc.doc_type.value, "text": doc.text,
                "metadata": {"page_count": doc.metadata.page_count,
                             "word_count": doc.metadata.word_count,
                             "title": doc.metadata.title, "author": doc.metadata.author},
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
