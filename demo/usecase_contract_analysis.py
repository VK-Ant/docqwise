"""
Demo: Contract Analysis
========================
Extract parties, clauses, dates, obligations from contract PDFs.

Usage:
    python demo/usecase_contract_analysis.py
"""

from docqwise.readers.pdf_reader import PDFReader
from docqwise.core.field import FieldResult, ExtractionResult
from docqwise.core.element import Entity, Relation
import re


def extract_contract_fields(pdf_path: str) -> dict:
    """Extract structured fields from a contract PDF."""

    reader = PDFReader()
    doc = reader.read(pdf_path)
    text = doc.text

    print(f"Reading: {pdf_path}")
    print(f"  Pages: {doc.metadata.page_count}, Words: {doc.metadata.word_count}")
    print()

    fields = {}

    # Agreement date
    date_match = re.search(r"Agreement\s*Date[:\s]*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if date_match:
        fields["agreement_date"] = FieldResult(
            value=date_match.group(1).strip(), confidence=0.97,
            extraction_method="regex",
        )

    # Parties
    parties = []
    party_patterns = [
        r'"Provider"\)\s*and\s*(.+?)\s*\("Client"\)',
        r'between\s+(.+?)\s*\("Provider"\)',
    ]
    for pattern in party_patterns:
        match = re.search(pattern, text)
        if match:
            parties.append(match.group(1).strip())

    if parties:
        fields["parties"] = FieldResult(
            value=parties, confidence=0.94,
            extraction_method="regex",
        )

    # Term
    term_match = re.search(r"period\s+of\s+(.+?)\s+(?:months|years)", text, re.IGNORECASE)
    if term_match:
        fields["term"] = FieldResult(
            value=term_match.group(1).strip() + " months", confidence=0.95,
            extraction_method="text_match",
        )

    # Compensation
    comp_match = re.search(r"total\s+amount\s+of\s+(INR\s*[\d,\.]+)", text, re.IGNORECASE)
    if comp_match:
        fields["compensation"] = FieldResult(
            value=comp_match.group(1).strip(), confidence=0.98,
            extraction_method="regex",
        )

    # Liability cap
    liability_match = re.search(r"liability.*?shall\s+not\s+exceed\s*\n?\s*(INR\s*[\d,\.]+)", text, re.IGNORECASE)
    if liability_match:
        fields["liability_cap"] = FieldResult(
            value=liability_match.group(1).strip(), confidence=0.96,
            extraction_method="regex",
        )

    # Governing law
    law_match = re.search(r"governed\s+by\s+the\s+laws\s+of\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
    if law_match:
        fields["governing_law"] = FieldResult(
            value=law_match.group(1).strip(), confidence=0.99,
            extraction_method="text_match",
        )

    return fields, text


def extract_entities(text: str) -> list[Entity]:
    """Extract named entities from contract text."""
    entities = []

    # Organizations
    org_patterns = [
        r"(?:between|and)\s+([A-Z][A-Za-z\s]+(?:Corporation|Solutions|LLC|Inc|Ltd))",
    ]
    for pattern in org_patterns:
        for match in re.finditer(pattern, text):
            entities.append(Entity(
                text=match.group(1).strip(),
                entity_type="ORG",
                confidence=0.95,
            ))

    # Money amounts
    for match in re.finditer(r"INR\s*[\d,\.]+", text):
        entities.append(Entity(
            text=match.group(0),
            entity_type="MONEY",
            confidence=0.98,
        ))

    # Dates
    for match in re.finditer(r"(?:August|September|October|November|December|January|February|March|April|May|June|July)\s+\d{1,2},\s*\d{4}", text):
        entities.append(Entity(
            text=match.group(0),
            entity_type="DATE",
            confidence=0.96,
        ))

    return entities


def extract_relations(entities: list[Entity], text: str) -> list[Relation]:
    """Extract relations between entities."""
    relations = []

    orgs = [e for e in entities if e.entity_type == "ORG"]
    money = [e for e in entities if e.entity_type == "MONEY"]
    dates = [e for e in entities if e.entity_type == "DATE"]

    if len(orgs) >= 2:
        relations.append(Relation(
            subject=orgs[0].text,
            predicate="provides_service_to",
            object=orgs[1].text,
            confidence=0.90,
        ))

    if orgs and money:
        relations.append(Relation(
            subject=orgs[-1].text if len(orgs) > 1 else orgs[0].text,
            predicate="shall_pay",
            object=money[0].text,
            confidence=0.93,
        ))

    if dates:
        relations.append(Relation(
            subject="Agreement",
            predicate="commences_on",
            object=dates[0].text,
            confidence=0.97,
        ))

    if "laws of India" in text.lower() or "governed by" in text.lower():
        relations.append(Relation(
            subject="Agreement",
            predicate="governed_by",
            object="laws of India",
            confidence=0.98,
        ))

    return relations


def main():
    print("=" * 60)
    print("DEMO: Contract Analysis")
    print("=" * 60)
    print()

    # Extract fields
    fields, text = extract_contract_fields("demo/sample_contract.pdf")

    print("Contract Fields:")
    print("-" * 60)
    for name, field in fields.items():
        print(f"  {name:20s} = {str(field.value):30s} [{field.confidence:.2f}]")

    # Extract entities
    print(f"\nEntities:")
    print("-" * 60)
    entities = extract_entities(text)
    for e in entities:
        print(f"  [{e.entity_type:8s}] {e.text}")

    # Extract relations
    print(f"\nRelations:")
    print("-" * 60)
    relations = extract_relations(entities, text)
    for r in relations:
        print(f"  {r.subject} --[{r.predicate}]--> {r.object}")

    # Build extraction result
    result = ExtractionResult(
        doc_id="contract_001",
        source_path="demo/sample_contract.pdf",
        fields=fields,
        confidence=sum(f.confidence for f in fields.values()) / len(fields),
        strategy_used="fast",
    )

    print(f"\nJSON output:\n{result.to_json()}")
    print("\nDone!")


if __name__ == "__main__":
    main()
