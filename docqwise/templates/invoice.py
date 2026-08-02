"""Invoice extraction template — LLM-first, regex fallback."""
from __future__ import annotations
from docqwise.core.document import DocqwiseDocument
from docqwise.core.field import ExtractionResult

class InvoiceTemplate:
    SCHEMA = {
        "invoice_number": {"type": "string", "description": "Invoice number or ID"},
        "date": {"type": "date", "description": "Invoice date"},
        "due_date": {"type": "date", "description": "Payment due date"},
        "vendor_name": {"type": "string", "description": "Vendor or seller company name"},
        "client_name": {"type": "string", "description": "Client or buyer company name"},
        "line_items": {"type": "array", "description": "List of items with description, quantity, unit price, total"},
        "subtotal": {"type": "number", "description": "Subtotal before tax"},
        "tax": {"type": "number", "description": "Tax amount"},
        "total": {"type": "number", "description": "Total amount due"},
        "po_number": {"type": "string", "description": "Purchase order number"},
        "payment_terms": {"type": "string", "description": "Payment terms (e.g. Net 30)"},
        "bank_details": {"type": "string", "description": "Bank account details for payment"},
    }

    def extract(self, document: DocqwiseDocument) -> ExtractionResult:
        from docqwise.extractors.llm_extractor import LLMFieldExtractor
        return LLMFieldExtractor().extract_fields(document, schema=self.SCHEMA)
