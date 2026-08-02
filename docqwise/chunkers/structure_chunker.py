"""Structure-preserving chunker (default). Respects tables, headings, lists."""
from __future__ import annotations
import hashlib, re
from docqwise.chunkers.base import BaseChunker
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk, ChunkMetadata, ElementType

class StructureChunker(BaseChunker):
    def __init__(self, max_tokens: int = 512, overlap: int = 50,
                 preserve_tables: bool = True, attach_headings: bool = True):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.preserve_tables = preserve_tables
        self.attach_headings = attach_headings

    def chunk(self, document: DocqwiseDocument, **kwargs) -> list[DocqwiseChunk]:
        text = document.text
        if not text.strip():
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_heading = ""
        current_text = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if self._is_heading(para):
                if current_text.strip():
                    chunks.extend(self._split_if_needed(current_text, document.doc_id,
                                                         current_heading, len(chunks)))
                current_heading = para
                current_text = ""
            elif self.preserve_tables and self._is_table(para):
                if current_text.strip():
                    chunks.extend(self._split_if_needed(current_text, document.doc_id,
                                                         current_heading, len(chunks)))
                    current_text = ""
                chunk_id = hashlib.md5(para.encode()).hexdigest()[:12]
                chunks.append(DocqwiseChunk(
                    chunk_id=chunk_id, text=para,
                    metadata=ChunkMetadata(doc_id=document.doc_id, chunk_index=len(chunks),
                                           element_type=ElementType.TABLE,
                                           heading_context=current_heading),
                ))
            else:
                if len((current_text + " " + para).split()) > self.max_tokens:
                    if current_text.strip():
                        chunks.extend(self._split_if_needed(current_text, document.doc_id,
                                                             current_heading, len(chunks)))
                    current_text = para
                else:
                    current_text = (current_text + "\n\n" + para).strip()
        if current_text.strip():
            chunks.extend(self._split_if_needed(current_text, document.doc_id,
                                                 current_heading, len(chunks)))
        return chunks

    def _split_if_needed(self, text: str, doc_id: str, heading: str, start_idx: int) -> list[DocqwiseChunk]:
        words = text.split()
        if len(words) <= self.max_tokens:
            chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
            return [DocqwiseChunk(
                chunk_id=chunk_id, text=text,
                metadata=ChunkMetadata(doc_id=doc_id, chunk_index=start_idx,
                                       element_type=ElementType.TEXT, heading_context=heading),
            )]
        chunks = []
        for i in range(0, len(words), self.max_tokens - self.overlap):
            chunk_words = words[i:i + self.max_tokens]
            chunk_text = " ".join(chunk_words)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append(DocqwiseChunk(
                chunk_id=chunk_id, text=chunk_text,
                metadata=ChunkMetadata(doc_id=doc_id, chunk_index=start_idx + len(chunks),
                                       element_type=ElementType.TEXT, heading_context=heading),
            ))
        return chunks

    def _is_heading(self, text: str) -> bool:
        lines = text.strip().split("\n")
        if len(lines) != 1:
            return False
        line = lines[0].strip()
        if line.startswith("#"):
            return True
        if line.isupper() and len(line.split()) <= 8:
            return True
        return False

    def _is_table(self, text: str) -> bool:
        lines = text.strip().split("\n")
        pipe_lines = sum(1 for l in lines if l.count("|") >= 2)
        return pipe_lines >= 2
