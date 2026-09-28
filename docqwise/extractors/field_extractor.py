"""Field extraction using regex patterns and schema matching."""
from __future__ import annotations
import re
from datetime import datetime
from docqwise.extractors.base import BaseFieldExtractor
from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import FieldResult, ExtractionResult

class RegexFieldExtractor(BaseFieldExtractor):
    """Extract fields using regex patterns. Deterministic, no ML."""

    COMMON_PATTERNS = {
        "date": [r"\d{4}-\d{2}-\d{2}", r"\d{2}/\d{2}/\d{4}",
                 r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}"],
        "email": [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
        "phone": [r"\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}"],
        "url": [r"https?://[^\s<>\"']+"],
        "currency": [r"(?:USD|INR|EUR|GBP|JPY|\$|Rs\.?|€|£)\s*[\d,]+\.?\d*"],
        "percentage": [r"\d+\.?\d*\s*%"],
    }

    def extract_fields(self, document: DocqwiseDocument, schema: dict = None,
                       prompt: str = None, prompt_template: str = None) -> ExtractionResult:
        text = document.text
        fields = {}

        if schema:
            for field_name, field_spec in schema.items():
                value = self._extract_by_spec(text, field_name, field_spec)
                if value is not None:
                    fields[field_name] = value
        else:
            fields = self._auto_extract(text)

        confidence = sum(f.confidence for f in fields.values()) / len(fields) if fields else 0.0
        return ExtractionResult(
            doc_id=document.doc_id, source_path=document.source_path,
            fields=fields, confidence=confidence, strategy_used="regex",
        )

    def _extract_by_spec(self, text: str, name: str, spec: dict) -> FieldResult | None:
        field_type = spec.get("type", "string")
        pattern = spec.get("pattern", None)

        if pattern:
            match = re.search(pattern, text)
            if match:
                return FieldResult(value=match.group(0), confidence=0.95, extraction_method="regex_pattern")

        label_pattern = rf"(?:{name.replace('_', '[_ ]')})\s*[:\-=]\s*(.+?)(?:\n|$)"
        match = re.search(label_pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if field_type == "number":
                try:
                    value = float(re.sub(r"[,$]", "", value))
                except ValueError:
                    pass
            elif field_type == "date":
                pass  # keep as string
            return FieldResult(value=value, confidence=0.90, raw_text=match.group(0), extraction_method="label_match")
        return None

    def _auto_extract(self, text: str) -> dict[str, FieldResult]:
        fields = {}
        for field_type, patterns in self.COMMON_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    key = f"{field_type}_{len(fields)}" if field_type in [f.split("_")[0] for f in fields] else field_type
                    fields[key] = FieldResult(
                        value=matches[0] if len(matches) == 1 else matches,
                        confidence=0.85, extraction_method="auto_regex",
                    )
                    break
        # Named field detection
        for match in re.finditer(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s*[:\-]\s*(.+?)(?:\n|$)", text):
            label = match.group(1).strip().lower().replace(" ", "_")
            value = match.group(2).strip()
            if label not in fields and len(label) < 30:
                fields[label] = FieldResult(value=value, confidence=0.80, raw_text=match.group(0), extraction_method="label_detect")
        return fields
