"""Zero-shot document classifier."""
from __future__ import annotations
import re
from collections import Counter

class ZeroShotClassifier:
    KEYWORDS = {
        "invoice": ["invoice", "bill", "amount due", "payment", "total", "subtotal", "tax", "qty"],
        "contract": ["agreement", "parties", "whereas", "hereby", "shall", "governing law", "term", "liability"],
        "resume": ["experience", "education", "skills", "objective", "certifications", "references"],
        "receipt": ["receipt", "thank you", "paid", "change", "cash", "card"],
        "report": ["report", "analysis", "findings", "conclusion", "summary", "methodology"],
        "email": ["from:", "to:", "subject:", "date:", "dear", "regards", "sincerely"],
        "research_paper": ["abstract", "introduction", "methodology", "results", "discussion", "references", "doi"],
    }

    def classify(self, text: str, labels: list[str] = None) -> list[dict]:
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        word_freq = Counter(words)
        candidates = labels or list(self.KEYWORDS.keys())
        scores = []
        for label in candidates:
            keywords = self.KEYWORDS.get(label, [label])
            score = sum(text_lower.count(kw) for kw in keywords)
            score_norm = score / max(len(words), 1) * 100
            scores.append({"label": label, "score": score, "confidence": min(score_norm, 1.0)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        total = sum(s["score"] for s in scores) or 1
        for s in scores:
            s["confidence"] = round(s["score"] / total, 3)
        return scores
