"""Natural language Q&A engine with source attribution.

Every answer tells you WHICH document it came from.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Optional

from docqwise.retrieval.query_router import QueryRouter


class QAAnswer:
    """Q&A result with source attribution."""

    def __init__(self, answer: str, source: str = "", confidence: float = 0.0,
                 method: str = ""):
        self.answer = answer
        self.source = source
        self.source_name = os.path.basename(source) if source else ""
        self.confidence = confidence
        self.method = method  # computation | semantic | llm | direct

    def __str__(self):
        if self.source_name:
            return f"{self.answer}\n  Source: {self.source_name}"
        return self.answer

    def __repr__(self):
        return f"QAAnswer(answer='{self.answer[:50]}', source='{self.source_name}')"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "source": self.source,
            "source_name": self.source_name,
            "confidence": self.confidence,
            "method": self.method,
        }


class QAEngine:
    def __init__(self, vector_store=None, embedder=None):
        self._store = vector_store
        self._embedder = embedder
        self._router = QueryRouter()
        self._structured_data = {}

    def register_structured(self, source: str, headers: list[str], rows: list[dict]):
        self._structured_data[source] = {"headers": headers, "rows": rows}

    def ask(self, question: str, source: str = None) -> str:
        result = self.ask_with_source(question, source)
        return str(result)

    def ask_with_source(self, question: str, source: str = None) -> QAAnswer:
        """Answer question and return source attribution."""
        if source:
            return self._ask_source(question, source)

        route = self._router.route(question)

        # Computation on structured data
        if route == "computation" and self._structured_data:
            best_source = self._find_best_structured_source(question)
            if best_source:
                answer = self._compute(question, best_source)
                return QAAnswer(answer=answer, source=best_source,
                                confidence=1.0, method="computation")

        if route == "filter" and self._structured_data:
            best_source = self._find_best_structured_source(question)
            if best_source:
                answer = self._filter(question, best_source)
                return QAAnswer(answer=answer, source=best_source,
                                confidence=0.9, method="filter")

        # Search all documents
        return self._search_and_answer(question)

    def _ask_source(self, question: str, source: str) -> QAAnswer:
        route = self._router.route(question)
        matched_key = self._match_structured_source(source)

        if matched_key and route in ("computation", "filter"):
            if route == "computation":
                answer = self._compute(question, matched_key)
            else:
                answer = self._filter(question, matched_key)
            return QAAnswer(answer=answer, source=source,
                            confidence=1.0, method=route)

        return self._search_and_answer(question, source_filter=source)

    def _match_structured_source(self, source: str) -> Optional[str]:
        if source in self._structured_data:
            return source
        source_name = os.path.basename(source).lower()
        for key in self._structured_data:
            key_name = os.path.basename(key).lower()
            if source_name in key_name or key_name in source_name:
                return key
        return None

    def _find_best_structured_source(self, question: str) -> Optional[str]:
        if len(self._structured_data) == 1:
            return list(self._structured_data.keys())[0]
        q = question.lower()
        best_source, best_score = None, 0
        for source, data in self._structured_data.items():
            score = 0
            name = os.path.basename(source).lower().split(".")[0]
            if name in q:
                score += 10
            for header in data["headers"]:
                if header.lower() in q:
                    score += 5
            if score > best_score:
                best_score = score
                best_source = source
        return best_source or list(self._structured_data.keys())[0]

    def _search_and_answer(self, question: str, source_filter: str = None) -> QAAnswer:
        chunks_text = None
        found_source = source_filter or ""

        # Try semantic search
        if self._embedder and self._store:
            try:
                results = self._search_chunks(question, source_filter=source_filter)
                if results:
                    chunks_text = "\n\n".join(r["text"] for r in results)
                    found_source = results[0].get("source_path", source_filter or "")
            except Exception:
                pass

        # Fallback: read source file directly
        if not chunks_text and source_filter:
            chunks_text = self._read_source_text(source_filter)
            found_source = source_filter

        if not chunks_text:
            return QAAnswer(answer="No relevant documents found.", method="none")

        # Send to LLM
        llm_answer = self._answer_with_llm(question, chunks_text)
        if llm_answer:
            return QAAnswer(answer=llm_answer, source=found_source,
                            confidence=0.85, method="llm")

        return QAAnswer(answer=chunks_text, source=found_source,
                        confidence=0.5, method="direct")

    def _search_chunks(self, question: str, source_filter: str = None) -> list[dict]:
        query_emb = self._embedder.embed(question)
        filters = None
        if source_filter:
            filters = {"source_path": source_filter}
        return self._store.search(query_emb, top_k=5, filters=filters)

    def _read_source_text(self, source: str) -> Optional[str]:
        try:
            if not os.path.exists(source):
                return None
            from docqwise.readers.auto_reader import AutoReader
            doc = AutoReader().read(source)
            return doc.text if doc.text else None
        except Exception:
            return None

    def _answer_with_llm(self, question: str, context: str) -> Optional[str]:
        try:
            from docqwise.factory import LLMFactory
            llm = LLMFactory.get()
            if llm is None:
                return None
            prompt = (
                f"Answer this question based on the document below.\n"
                f"Be concise. Give exact values. One paragraph max.\n\n"
                f"Question: {question}\n\n"
                f"Document:\n{context}\n\n"
                f"Answer:"
            )
            return llm.generate(prompt).strip()
        except Exception:
            return None

    def _compute(self, question: str, source: str = None) -> str:
        data = self._get_data(source)
        if not data:
            return "No structured data available."
        rows = data["rows"]
        q = question.lower()
        numeric_cols = self._find_numeric_columns(data)
        if not numeric_cols:
            return "No numeric columns found."
        col = numeric_cols[0]
        for nc in numeric_cols:
            if nc.lower() in q:
                col = nc
                break
        values = [float(str(r[col]).replace(",", "")) for r in rows
                  if self._is_number(r.get(col, ""))]
        if not values:
            return f"No numeric values in column '{col}'."
        if any(w in q for w in ["total", "sum"]):
            return f"{sum(values):,.2f}"
        elif any(w in q for w in ["average", "avg", "mean"]):
            return f"{sum(values)/len(values):,.2f}"
        elif any(w in q for w in ["count", "how many"]):
            return str(len(rows))
        elif any(w in q for w in ["max", "maximum", "highest", "largest"]):
            max_val = max(values)
            max_row = [r for r in rows if self._is_number(r.get(col, ""))
                       and float(str(r[col]).replace(",", "")) == max_val]
            if max_row:
                details = ", ".join(f"{k}={v}" for k, v in max_row[0].items() if k != col)
                return f"{max_val:,.2f} ({details})"
            return f"{max_val:,.2f}"
        elif any(w in q for w in ["min", "minimum", "smallest", "lowest"]):
            return f"{min(values):,.2f}"
        return f"Column '{col}': sum={sum(values):,.2f}, avg={sum(values)/len(values):,.2f}, count={len(values)}"

    def _filter(self, question: str, source: str = None) -> str:
        data = self._get_data(source)
        if not data:
            return "No structured data available."
        rows = data["rows"]
        q = question.lower()
        for header in data["headers"]:
            for row in rows:
                val = str(row.get(header, "")).lower()
                if val and val in q:
                    filtered = [r for r in rows if str(r.get(header, "")).lower() == val]
                    if filtered:
                        lines = [", ".join(f"{k}: {v}" for k, v in r.items()) for r in filtered]
                        return f"Found {len(filtered)} results:\n" + "\n".join(lines)
        return f"Total rows: {len(rows)}"

    def _get_data(self, source: str = None) -> dict | None:
        if source:
            matched = self._match_structured_source(source)
            if matched:
                return self._structured_data[matched]
        if self._structured_data:
            return list(self._structured_data.values())[0]
        return None

    def _find_numeric_columns(self, data: dict) -> list[str]:
        numeric = []
        for header in data["headers"]:
            vals = [r.get(header, "") for r in data["rows"][:5]]
            if all(self._is_number(v) for v in vals if v):
                numeric.append(header)
        return numeric

    @staticmethod
    def _is_number(val) -> bool:
        try:
            float(str(val).replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False
