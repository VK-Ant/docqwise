"""Data quality scoring."""
from __future__ import annotations

class QualityScorer:
    def score(self, headers: list[str], rows: list[dict]) -> dict:
        if not rows or not headers:
            return {"completeness": 0, "consistency": 0, "uniqueness": 0, "overall": 0}
        total_cells = len(headers) * len(rows)
        filled = sum(1 for r in rows for h in headers if r.get(h))
        completeness = filled / total_cells if total_cells else 0
        unique_rows = len(set(tuple(sorted(r.items())) for r in rows))
        uniqueness = unique_rows / len(rows) if rows else 0
        type_consistent = 0
        for h in headers:
            vals = [type(r.get(h)).__name__ for r in rows if r.get(h)]
            if vals and len(set(vals)) == 1:
                type_consistent += 1
        consistency = type_consistent / len(headers) if headers else 0
        overall = (completeness + consistency + uniqueness) / 3
        return {"completeness": round(completeness, 3), "consistency": round(consistency, 3),
                "uniqueness": round(uniqueness, 3), "overall": round(overall, 3)}
