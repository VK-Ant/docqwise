"""RAG Strategy Selector.

User classifies document first, then chooses RAG strategy:
  - GeneralRAG:     chunk → embed → retrieve → LLM (default)
  - GraphRAG:       extract entities → build graph → graph-enhanced retrieval → LLM
  - MultimodalRAG:  text + images → vision LLM → combined retrieval

Usage:
    from docqwise.retrieval.rag_strategy import RAGStrategy

    strategy = RAGStrategy(dq, mode="graphrag")
    answer = strategy.ask("Who is the vendor for PO-2012?")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("docqwise")


class RAGStrategy:
    """Unified RAG strategy selector."""

    MODES = ["general", "graphrag", "multimodal"]

    def __init__(self, engine=None, mode: str = "general"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown RAG mode: {mode}. Available: {self.MODES}")
        self.mode = mode
        self._engine = engine

    def ask(self, question: str, source: str = None, prompt: str = None, **kwargs) -> dict:
        """Ask using selected RAG strategy. Returns answer + evidence."""
        if self.mode == "graphrag":
            return self._graphrag_ask(question, source, prompt=prompt, **kwargs)
        elif self.mode == "multimodal":
            return self._multimodal_ask(question, source, prompt=prompt, **kwargs)
        else:
            return self._general_ask(question, source, prompt=prompt, **kwargs)

    def _general_ask(self, question, source, prompt=None, **kwargs):
        """Standard RAG: chunk → embed → retrieve → LLM."""
        if prompt and source:
            # Custom prompt with specific source
            result = self._engine.extract_fields(source, prompt=prompt)
            fields_text = ", ".join(f"{k}: {v.value}" for k, v in result.fields.items())
            return {
                "answer": fields_text or "No fields extracted.",
                "source": source.split("/")[-1].split("\\")[-1],
                "method": "general_rag",
                "confidence": result.confidence,
                "evidence": [],
            }
        result = self._engine.ask_with_source(question, source=source)
        return {
            "answer": result.answer,
            "source": result.source_name,
            "method": "general_rag",
            "confidence": result.confidence,
            "evidence": [],
        }

    def _graphrag_ask(self, question, source, prompt=None, **kwargs):
        """GraphRAG: extract entities → build graph → graph-enhanced retrieval."""
        from docqwise.retrieval.graphrag import GraphRAGEngine

        graphrag = GraphRAGEngine(self._engine)

        # Build graph if not already built
        if not hasattr(self._engine, '_doc_graph') or self._engine._doc_graph is None:
            graphrag.build_from_ingested()

        # Query using graph
        result = graphrag.query(question, source=source, prompt=prompt)
        return result

    def _multimodal_ask(self, question, source, prompt=None, **kwargs):
        """MultimodalRAG: text + images → combined retrieval."""
        # Extract text answer
        text_result = self._engine.ask_with_source(question, source=source)

        # If source is image-heavy, use vision
        if source and source.lower().endswith(('.jpg', '.png', '.tiff', '.jpeg')):
            from docqwise.extractors.multimodal_extractor import MultimodalExtractor
            vis = MultimodalExtractor(model=kwargs.get("model", "gpt-4o-mini"))
            vis_result = vis.extract_from_image(source, prompt=question)
            return {
                "answer": text_result.answer,
                "source": text_result.source_name,
                "method": "multimodal_rag",
                "confidence": text_result.confidence,
                "vision_fields": vis_result.to_dict() if vis_result.fields else {},
                "evidence": [],
            }

        return {
            "answer": text_result.answer,
            "source": text_result.source_name,
            "method": "multimodal_rag",
            "confidence": text_result.confidence,
            "evidence": [],
        }
