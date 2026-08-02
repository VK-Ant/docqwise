"""Routes queries to computation engine or semantic search."""
from __future__ import annotations
import re

class QueryRouter:
    COMPUTATION_KEYWORDS = ["total", "sum", "average", "avg", "count", "how many",
                            "maximum", "max", "minimum", "min", "mean", "median"]
    FILTER_KEYWORDS = ["show me", "list", "filter", "which", "where", "overdue",
                       "pending", "find all", "display"]

    def route(self, query: str) -> str:
        q = query.lower().strip()
        for kw in self.COMPUTATION_KEYWORDS:
            if kw in q:
                return "computation"
        for kw in self.FILTER_KEYWORDS:
            if kw in q:
                return "filter"
        return "semantic"
