"""PII detection and redaction."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class PIIMatch:
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 0.9

class PIIDetector:
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    }

    def detect(self, text: str, types: list[str] = None) -> list[PIIMatch]:
        matches = []
        for pii_type, pattern in self.PATTERNS.items():
            if types and pii_type not in types:
                continue
            for match in re.finditer(pattern, text):
                matches.append(PIIMatch(pii_type=pii_type, value=match.group(),
                                         start=match.start(), end=match.end()))
        return matches

    def redact(self, text: str, types: list[str] = None, replacement: str = "[REDACTED]") -> str:
        matches = sorted(self.detect(text, types), key=lambda m: m.start, reverse=True)
        result = text
        for match in matches:
            result = result[:match.start] + replacement + result[match.end:]
        return result
