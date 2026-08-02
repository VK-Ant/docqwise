"""Deterministic strategy router — picks the best extraction method per document.

Rules-based. No ML. But smart about when to use which method.
"""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument


class StrategyRouter:
    """Routes documents to the optimal extraction strategy.
    
    Strategies:
        fast       — regex only, no ML, instant (digital PDFs with clean text)
        ocr        — OCR + regex (scanned documents, images)
        llm        — LLM-based extraction (complex layouts, mixed content)
        vision     — multimodal vision LLM (handwriting, degraded scans, charts)
        structured — schema engine (CSV, JSON, Excel, databases)
    """

    def __init__(self, mode: str = "auto", confidence_threshold: float = 0.85):
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def route(self, document: DocqwiseDocument) -> str:
        """Pick the best strategy for this document."""
        if self.mode != "auto":
            return self.mode

        # Structured data — no extraction needed, just schema
        if document.doc_type.value in ("csv", "json", "xml", "yaml", "excel", "parquet"):
            return "structured"

        # Database — SQL query engine
        if document.doc_type.value == "database":
            return "database"

        # Scanned/image documents — need OCR + potentially vision LLM
        if document.metadata.is_scanned or not document.metadata.has_selectable_text:
            if document.doc_type.value == "image":
                return "vision"  # pure image → vision LLM best
            return "ocr"  # scanned PDF → OCR first, then LLM

        # Digital documents with selectable text
        if document.metadata.has_selectable_text:
            word_count = document.metadata.word_count
            if word_count < 50:
                return "fast"  # very short doc → regex is fine
            if word_count > 5000:
                return "llm"  # long doc → LLM understands context better
            # Medium documents — LLM preferred but regex works
            return "llm"

        return "llm"  # default to LLM

    def describe(self, strategy: str) -> str:
        """Describe what a strategy does."""
        descriptions = {
            "fast": "Regex patterns — instant, no ML, works on clean digital text",
            "ocr": "OCR engine → text extraction → field matching",
            "llm": "LLM-based — sends text to AI model for intelligent extraction",
            "vision": "Vision LLM — renders page as image, uses multimodal AI",
            "structured": "Schema engine — auto-detect types, SQL queries, validation",
            "database": "Database connector — direct SQL queries",
        }
        return descriptions.get(strategy, "Unknown strategy")
