"""CSV exporter."""
from __future__ import annotations
import csv, os
from docqwise.core.document import DocqwiseDocument

class CSVExporter:
    def export(self, documents: list[DocqwiseDocument], path: str, **kwargs):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "source_path", "doc_type", "pages", "words", "title"])
            for doc in documents:
                writer.writerow([doc.doc_id, doc.source_path, doc.doc_type.value,
                                doc.metadata.page_count, doc.metadata.word_count, doc.metadata.title])
