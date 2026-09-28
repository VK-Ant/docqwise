"""Demo 05: Agentic Document Extraction (v0.4.0)

Multi-pass, self-correcting extraction with validation, retry, and cross-check.

Usage:
    python demo/05_agentic.py

Requirements:
    pip install docqwise
    # For LLM extraction: ollama pull nemotron-mini
"""

from docqwise import Docqwise
from docqwise.agent import ExtractionAgent
from docqwise.agent.extraction_agent import ValidationRule


def demo_basic_agentic():
    """Basic agentic extraction via engine."""
    print("=" * 60)
    print("DEMO: Basic Agentic Extraction")
    print("=" * 60)

    dq = Docqwise()

    # Agentic extraction with template
    result = dq.extract_agentic(
        "demo/sample_invoice.pdf",
        template="invoice",
        max_retries=2,
        cross_check=True,
    )

    print(f"\nExtracted {len(result.fields)} fields:")
    for name, field in result.fields.items():
        print(f"  {name}: {field.value} (confidence: {field.confidence:.2f}, "
              f"method: {field.extraction_method})")

    print(f"\nOverall confidence: {result.confidence:.2f}")
    print(f"Strategy: {result.strategy_used}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def demo_agent_with_trace():
    """Direct agent usage with full audit trace."""
    print("\n" + "=" * 60)
    print("DEMO: Agent with Audit Trace")
    print("=" * 60)

    agent = ExtractionAgent(
        max_retries=3,
        cross_check=True,
        confidence_threshold=0.7,
    )

    result = agent.extract(
        "demo/sample_invoice.pdf",
        template="invoice",
    )

    print(f"\nExtracted {len(result.fields)} fields")
    print(f"Processing time: {result.processing_time_ms:.0f}ms")
    print(f"\nFull trace:")
    print(agent.trace.summary())


def demo_custom_validation():
    """Custom validation rules for domain-specific checks."""
    print("\n" + "=" * 60)
    print("DEMO: Custom Validation Rules")
    print("=" * 60)

    rules = [
        ValidationRule("invoice_number", "required"),
        ValidationRule("total_amount", "required"),
        ValidationRule("total_amount", "range", {"min": 0, "max": 10000000}),
        ValidationRule("date", "type", {"expected": "date"}),
        ValidationRule("email", "regex", {"pattern": r"[\w.]+@[\w.]+\.\w+"}),
    ]

    agent = ExtractionAgent(
        validation_rules=rules,
        max_retries=2,
    )

    schema = {
        "invoice_number": "string",
        "total_amount": "number",
        "date": "date",
        "vendor_name": "string",
        "email": "string",
    }

    result = agent.extract("demo/sample_invoice.pdf", schema=schema)

    print(f"\nExtracted fields:")
    for name, field in result.fields.items():
        print(f"  {name}: {field.value}")

    print(f"\nValidation: {len(agent.trace.retried_fields)} fields needed retry")
    print(f"Confidence: {result.confidence:.2f}")


def demo_custom_prompt():
    """Agentic extraction with custom prompts."""
    print("\n" + "=" * 60)
    print("DEMO: Custom Prompt + Agentic")
    print("=" * 60)

    dq = Docqwise()

    result = dq.extract_agentic(
        "demo/sample_contract.pdf",
        schema={
            "parties": "string",
            "effective_date": "date",
            "termination_clause": "string",
            "governing_law": "string",
        },
        system_prompt="You are a legal document analyst. Extract key contract terms precisely.",
    )

    print(f"\nContract extraction ({result.confidence:.2f} confidence):")
    for name, field in result.fields.items():
        val = str(field.value)[:80]
        print(f"  {name}: {val}")


if __name__ == "__main__":
    demo_basic_agentic()
    demo_agent_with_trace()
    demo_custom_validation()
    demo_custom_prompt()
    print("\n" + "=" * 60)
    print("All agentic demos complete!")
    print("=" * 60)
