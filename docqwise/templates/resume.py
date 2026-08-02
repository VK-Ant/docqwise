"""Resume extraction template — LLM-first, regex fallback."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import ExtractionResult

class ResumeTemplate:
    SCHEMA = {
        "name": {"type": "string", "description": "Full name of the candidate"},
        "email": {"type": "string", "description": "Email address"},
        "phone": {"type": "string", "description": "Phone number"},
        "location": {"type": "string", "description": "City, state, or country"},
        "summary": {"type": "string", "description": "Professional summary or objective"},
        "experience": {"type": "array", "description": "Work experience entries with company, role, dates, responsibilities"},
        "education": {"type": "array", "description": "Education entries with institution, degree, dates"},
        "skills": {"type": "array", "description": "List of technical and soft skills"},
        "certifications": {"type": "array", "description": "Professional certifications"},
        "languages": {"type": "array", "description": "Languages spoken"},
    }

    def extract(self, document: DocqwiseDocument) -> ExtractionResult:
        from docqwise.extractors.llm_extractor import LLMFieldExtractor
        return LLMFieldExtractor().extract_fields(document, schema=self.SCHEMA)
