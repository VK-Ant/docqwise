"""Entity extraction using regex and optional spaCy."""
from __future__ import annotations
import re
from docqwise.extractors.base import BaseEntityExtractor
from docqwise.core.document import DocqwiseDocument
from docqwise.core.element import Entity

class RegexEntityExtractor(BaseEntityExtractor):
    """Extract entities using regex patterns. No ML dependency."""

    PATTERNS = {
        "MONEY": [r"(?:USD|INR|EUR|GBP|\$|Rs\.?|€|£)\s*[\d,]+\.?\d*",
                  r"[\d,]+\.?\d*\s*(?:USD|INR|EUR|GBP|dollars|rupees)"],
        "DATE": [r"\d{4}-\d{2}-\d{2}",
                 r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}",
                 r"\d{1,2}/\d{1,2}/\d{4}"],
        "EMAIL": [r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"],
        "PHONE": [r"\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}"],
        "URL": [r"https?://[^\s<>\"']+"],
        "PERCENTAGE": [r"\d+\.?\d*\s*%"],
    }

    def extract_entities(self, document: DocqwiseDocument, types: list[str] = None) -> list[Entity]:
        text = document.text
        entities = []
        filter_types = set(t.upper() for t in types) if types else None

        for entity_type, patterns in self.PATTERNS.items():
            if filter_types and entity_type not in filter_types:
                continue
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entities.append(Entity(
                        text=match.group(0), entity_type=entity_type,
                        confidence=0.90,
                    ))

        # Try spaCy for ORG, PERSON, LOCATION
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:100000])  # limit for performance
            for ent in doc.ents:
                if filter_types and ent.label_ not in filter_types:
                    continue
                if ent.label_ in ("ORG", "PERSON", "GPE", "LOC", "FAC"):
                    mapped_type = "LOCATION" if ent.label_ in ("GPE", "LOC", "FAC") else ent.label_
                    entities.append(Entity(text=ent.text, entity_type=mapped_type, confidence=0.92))
        except (ImportError, OSError):
            pass

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = (e.text, e.entity_type)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique
