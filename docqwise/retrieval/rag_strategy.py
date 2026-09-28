"""RAG strategy — routes between GeneralRAG, GraphRAG, and MultimodalRAG.

Usage:
    strategy = RAGStrategy(mode="graphrag", llm=my_llm)
    result = strategy.query(question, sources=["doc1.pdf", "doc2.pdf"])
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("docqwise")


@dataclass
class QAAnswer:
    """Q&A result with source attribution."""
    answer: str
    source: str = ""
    source_name: str = ""
    confidence: float = 0.0
    method: str = ""
    evidence: list = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if not self.source_name and self.source:
            self.source_name = os.path.basename(self.source)

    def __str__(self):
        return self.answer

    def __repr__(self):
        return f"QAAnswer(answer='{self.answer[:50]}...', source='{self.source_name}', confidence={self.confidence:.2f})"


class RAGStrategy:
    """Routes queries to the right RAG mode.

    Modes:
        - general: standard semantic search + LLM
        - graphrag: entity graph traversal + LLM with evidence chains
        - multimodal: text + images combined
    """

    MODES = ("general", "graphrag", "multimodal")

    def __init__(self, mode: str = "general", llm=None, embedder=None):
        if mode not in self.MODES:
            raise ValueError(f"Unknown RAG mode: {mode}. Available: {self.MODES}")
        self.mode = mode
        self._llm = llm
        self._embedder = embedder

    def query(self, question: str, sources: list[str] = None,
              reader=None, vector_store=None,
              prompt: str = None, system_prompt: str = None) -> QAAnswer:
        """Route query to the right RAG engine."""
        if self.mode == "graphrag":
            return self._query_graphrag(question, sources, reader, prompt, system_prompt)
        elif self.mode == "multimodal":
            return self._query_multimodal(question, sources, reader, vector_store, prompt, system_prompt)
        else:
            return self._query_general(question, sources, reader, vector_store, prompt, system_prompt)

    def _query_general(self, question, sources, reader, vector_store, prompt, system_prompt) -> QAAnswer:
        """Standard RAG: retrieve relevant chunks → LLM answer."""
        llm = self._ensure_llm()
        if not llm:
            return QAAnswer(answer="No LLM available", confidence=0.0, method="general")

        # Get context from vector store
        context = ""
        source_path = ""
        if vector_store and self._embedder:
            results = vector_store.search(self._embedder.embed(question), top_k=5)
            if results:
                context = "\n\n".join(r["text"] for r in results)
                source_path = results[0].get("source_path", "")
        elif sources and reader:
            # Fallback: read source files directly
            texts = []
            for src in sources[:3]:
                try:
                    doc = reader.read(src)
                    texts.append(doc.text[:2000])
                    if not source_path:
                        source_path = src
                except Exception:
                    pass
            context = "\n\n".join(texts)

        if prompt:
            if "{context}" in prompt:
                final = prompt.replace("{context}", context)
            else:
                final = f"{prompt}\n\nContext:\n{context}"
        else:
            final = f"Answer this question using the context below.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"

        kwargs = {}
        if system_prompt:
            kwargs["system_prompt"] = system_prompt

        answer = llm.generate(final, **kwargs)
        return QAAnswer(
            answer=answer,
            source=source_path,
            confidence=0.85,
            method="general",
        )

    def _query_graphrag(self, question, sources, reader, prompt, system_prompt) -> QAAnswer:
        """GraphRAG: build entity graph → traverse → LLM with evidence."""
        from docqwise.retrieval.graphrag import GraphRAGEngine

        engine = GraphRAGEngine(llm=self._llm, embedder=self._embedder)

        if sources and reader:
            engine.build_from_ingested(sources, reader=reader)
        elif sources:
            engine.build_from_documents(sources)

        result = engine.query(question, prompt=prompt)
        return QAAnswer(
            answer=result.get("answer", ""),
            source=result.get("source", ""),
            source_name=result.get("source_name", ""),
            confidence=result.get("confidence", 0.0),
            method="graphrag",
            evidence=result.get("evidence", []),
        )

    def _query_multimodal(self, question, sources, reader, vector_store, prompt, system_prompt) -> QAAnswer:
        """Multimodal RAG: text + images combined."""
        # Start with general RAG
        general = self._query_general(question, sources, reader, vector_store, prompt, system_prompt)

        # Try vision extraction on image/PDF sources
        if sources:
            from docqwise.extractors.multimodal_extractor import MultimodalExtractor
            for src in sources:
                if src.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf")):
                    try:
                        extractor = MultimodalExtractor(model="gpt-4o-mini")
                        result = extractor.extract_from_image(src) if not src.endswith(".pdf") \
                            else extractor.extract_from_pdf_page(src)
                        if result.fields:
                            vision_text = "\n".join(f"{k}: {v.value}" for k, v in result.fields.items())
                            general.answer = f"{general.answer}\n\nVisual extraction:\n{vision_text}"
                            general.method = "multimodal"
                    except Exception:
                        pass

        return general

    def _ensure_llm(self):
        if self._llm:
            return self._llm
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            from docqwise.llm.ollama import OllamaLLM
            self._llm = OllamaLLM(model="nemotron-mini")
            return self._llm
        except Exception:
            pass
        return None
