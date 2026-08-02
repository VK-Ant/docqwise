"""Fixed-size token window chunker."""
from __future__ import annotations
import hashlib
from docqwise.chunkers.base import BaseChunker
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, ElementType

class FixedChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, overlap: int = 50):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, document: DocqwiseDocument, **kwargs) -> list[DocqwiseChunk]:
        words = document.text.split()
        if not words:
            return []
        chunks = []
        step = max(1, self.max_tokens - self.overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.max_tokens]
            text = " ".join(chunk_words)
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            chunks.append(DocqwiseChunk(
                chunk_id=chunk_id, text=text,
                metadata=ChunkMetadata(doc_id=document.doc_id, chunk_index=len(chunks),
                                       element_type=ElementType.TEXT),
            ))
        return chunks
