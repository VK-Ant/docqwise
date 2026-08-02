"""Contract extraction template — LLM-first, regex fallback."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import ExtractionResult

class ContractTemplate:
    SCHEMA = {
        "agreement_date": {"type": "date", "description": "Date the agreement was signed or effective"},
        "parties": {"type": "array", "description": "Names of all parties to the agreement"},
        "effective_date": {"type": "date", "description": "When the agreement takes effect"},
        "term": {"type": "string", "description": "Duration of the agreement"},
        "scope_of_work": {"type": "string", "description": "Description of services or deliverables"},
        "compensation": {"type": "number", "description": "Total payment amount"},
        "payment_terms": {"type": "string", "description": "Payment schedule and conditions"},
        "liability_cap": {"type": "number", "description": "Maximum liability amount"},
        "termination_clause": {"type": "string", "description": "Conditions for early termination"},
        "governing_law": {"type": "string", "description": "Jurisdiction governing the agreement"},
        "confidentiality": {"type": "string", "description": "Confidentiality or NDA terms"},
    }

    def extract(self, document: DocqwiseDocument) -> ExtractionResult:
        from docqwise.extractors.llm_extractor import LLMFieldExtractor
        return LLMFieldExtractor().extract_fields(document, schema=self.SCHEMA)
