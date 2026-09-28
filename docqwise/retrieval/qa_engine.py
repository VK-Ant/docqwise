"""Natural language Q&A engine."""
from __future__ import annotations
import re
from collections import defaultdict
from docqwise.retrieval.query_router import QueryRouter

class QAEngine:
    def __init__(self, vector_store=None, embedder=None):
        self._store = vector_store
        self._embedder = embedder
        self._router = QueryRouter()
        self._structured_data = {}

    def register_structured(self, source: str, headers: list[str], rows: list[dict]):
        self._structured_data[source] = {"headers": headers, "rows": rows}

    def ask(self, question: str, source: str = None) -> str:
        route = self._router.route(question)
        if route == "computation" and self._structured_data:
            return self._compute(question, source)
        elif route == "filter" and self._structured_data:
            return self._filter(question, source)
        else:
            return self._semantic(question)

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
        values = [float(r[col]) for r in rows if self._is_number(r.get(col, ""))]
        if not values:
            return f"No numeric values in column '{col}'."
        if any(w in q for w in ["total", "sum"]):
            return f"{sum(values):,.2f}"
        elif any(w in q for w in ["average", "avg", "mean"]):
            return f"{sum(values)/len(values):,.2f}"
        elif any(w in q for w in ["count", "how many"]):
            return str(len(values))
        elif any(w in q for w in ["max", "maximum", "highest", "largest"]):
            max_val = max(values)
            max_row = [r for r in rows if self._is_number(r.get(col,"")) and float(r[col]) == max_val]
            if max_row:
                return f"{max_val:,.2f} ({', '.join(f'{k}={v}' for k,v in max_row[0].items() if k != col)})"
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

    def _semantic(self, question: str) -> str:
        if self._store and self._embedder:
            results = self._store.search(self._embedder.embed(question), top_k=3)
            if results:
                return "\n\n".join(r["text"] for r in results)
        return "No results found."

    def _get_data(self, source: str = None) -> dict | None:
        if source and source in self._structured_data:
            return self._structured_data[source]
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
