"""Receipt extraction template."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import ExtractionResult

class ReceiptTemplate:
    SCHEMA = {
        "merchant_name": {"type": "string", "description": "Store or merchant name"},
        "date": {"type": "date", "description": "Transaction date"},
        "items": {"type": "array", "description": "Purchased items with name, quantity, price"},
        "subtotal": {"type": "number", "description": "Subtotal before tax"},
        "tax": {"type": "number", "description": "Tax amount"},
        "total": {"type": "number", "description": "Total amount paid"},
        "payment_method": {"type": "string", "description": "Cash, card, UPI, etc."},
    }

    def extract(self, document: DocqwiseDocument) -> ExtractionResult:
        from docqwise.extractors.llm_extractor import LLMFieldExtractor
        return LLMFieldExtractor().extract_fields(document, schema=self.SCHEMA)
