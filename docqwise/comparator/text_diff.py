"""Document comparison using text diff."""
from __future__ import annotations
import difflib

class TextComparator:
    def compare(self, text_a: str, text_b: str) -> dict:
        lines_a = text_a.splitlines()
        lines_b = text_b.splitlines()
        differ = difflib.unified_diff(lines_a, lines_b, lineterm="")
        diff_lines = list(differ)
        additions = [l[1:] for l in diff_lines if l.startswith("+") and not l.startswith("+++")]
        deletions = [l[1:] for l in diff_lines if l.startswith("-") and not l.startswith("---")]
        similarity = difflib.SequenceMatcher(None, text_a, text_b).ratio()
        return {"additions": additions, "deletions": deletions,
                "additions_count": len(additions), "deletions_count": len(deletions),
                "similarity": round(similarity, 4), "diff": diff_lines}
