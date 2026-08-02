"""Sentence-level chunker."""
from __future__ import annotations
import hashlib, re
from docqwise.chunkers.base import BaseChunker
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, ElementType

class SentenceChunker(BaseChunker):
    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1):
        self.max_sentences = max_sentences
        self.overlap = overlap_sentences

    def chunk(self, document: DocqwiseDocument, **kwargs) -> list[DocqwiseChunk]:
        sentences = re.split(r'(?<=[.!?])\s+', document.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        chunks = []
        step = max(1, self.max_sentences - self.overlap)
        for i in range(0, len(sentences), step):
            group = sentences[i:i + self.max_sentences]
            text = " ".join(group)
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            chunks.append(DocqwiseChunk(
                chunk_id=chunk_id, text=text,
                metadata=ChunkMetadata(doc_id=document.doc_id, chunk_index=len(chunks),
                                       element_type=ElementType.TEXT),
            ))
        return chunks
