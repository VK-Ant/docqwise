"""Document metadata extraction."""
from docqwise.core.document import DocqwiseDocument

class MetadataExtractor:
    def extract(self, document: DocqwiseDocument) -> dict:
        return {
            "doc_id": document.doc_id,
            "source_path": document.source_path,
            "doc_type": document.doc_type.value,
            "page_count": document.metadata.page_count,
            "word_count": document.metadata.word_count,
            "language": document.metadata.language,
            "title": document.metadata.title,
            "author": document.metadata.author,
            "is_scanned": document.metadata.is_scanned,
            "file_hash": document.metadata.file_hash,
            "file_size": document.metadata.file_size_bytes,
        }
