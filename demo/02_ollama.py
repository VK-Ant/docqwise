"""
DocQWise Demo 2: Ollama (Local LLM)
=====================================

AI-powered extraction using Ollama running locally.
Zero cloud. Zero cost. Your data stays on your machine.

Requirements:
    pip install docqwise
    
    # Install and start Ollama: https://ollama.com
    ollama pull nemotron-mini
    ollama serve

Run:
    python demo/02_ollama.py
"""

import os
import tempfile
import shutil
from docqwise import Docqwise
from docqwise.llm.ollama import OllamaLLM


def main():
    print("=" * 60)
    print("DocQWise + Ollama — Local AI Document Extraction")
    print("=" * 60)
    print()

    store = os.path.join(tempfile.gettempdir(), "docqwise_ollama_demo")
    if os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)

    # Initialize Ollama LLM — change model name to match your ollama list
    MODEL = "nemotron-mini"  # or "qwen3.6" or "gemma4:12b"
    llm = OllamaLLM(model=MODEL)
    print(f"Using Ollama model: {MODEL}")
    print()

    dq = Docqwise(store_path=store)

    # ── Quick LLM test first ──
    print("0. LLM TEST")
    print("-" * 60)
    test_response = llm.generate(
        'Extract vendor and total from this text:\n'
        '"Invoice from Acme Corporation. Total amount: INR 2,18,300.00"\n'
        'Return JSON only: {"vendor": "...", "total": ...}'
    )
    print(f"   Raw LLM output: {test_response}")
    print()

    # ── 1. Invoice Extraction ──
    print("1. INVOICE EXTRACTION")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        template="invoice",
        model=MODEL,
    )
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Confidence: {result.confidence:.2f}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value):30s}")
    print()

    # ── 2. Contract Extraction ──
    print("2. CONTRACT EXTRACTION")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_contract.pdf",
        template="contract",
        model=MODEL,
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        val = str(field.value)[:40]
        print(f"   {name:20s} = {val}")
    print()

    # ── 3. Custom Schema Extraction ──
    print("3. CUSTOM SCHEMA (define your own fields)")
    print("-" * 60)
    schema = {
        "sender_company": {"type": "string", "description": "Company that sent the document"},
        "receiver_company": {"type": "string", "description": "Company receiving the document"},
        "total_value": {"type": "number", "description": "Total monetary value mentioned"},
        "key_dates": {"type": "array", "description": "All important dates in the document"},
        "document_purpose": {"type": "string", "description": "What is this document for"},
    }
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        schema=schema,
        model=MODEL,
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value):40s}")
    print()

    # ── 4. Auto Extract (no schema, no template — LLM decides) ──
    print("4. AUTO EXTRACT (LLM decides what to extract)")
    print("-" * 60)
    result = dq.auto_extract("demo/sample_invoice.pdf")
    print(f"   Strategy: {result.strategy_used}")
    for name, field in list(result.fields.items())[:8]:
        print(f"   {name:20s} = {str(field.value):30s}")
    print()

    # ── 5. Entity Extraction ──
    print("5. ENTITY EXTRACTION")
    print("-" * 60)
    entities = dq.extract_entities("demo/sample_contract.pdf")
    for e in entities[:8]:
        print(f"   [{e.entity_type:10s}] {e.text}")
    print()

    # ── 6. Custom Prompt — user designs the prompt ──
    print("6. CUSTOM PROMPT (you design, we run the pipeline)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        model=MODEL,
        prompt="""You are a financial document analyzer.
From this invoice, extract only these three things:
1. The company that SENT this invoice
2. The company that RECEIVED this invoice  
3. The exact total amount as a number

Document:
{context}

Return JSON: {"sender": "...", "receiver": "...", "amount": ...}
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {field.value}")
    print()

    # ── 7. System Prompt (set LLM role) ──
    print("7. SYSTEM PROMPT (set the LLM role)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        model=MODEL,
        system_prompt="You are a senior financial auditor at a Big 4 firm. Be precise with numbers. Flag any discrepancies.",
        prompt="""Review this invoice. Extract vendor, total, tax, and verify tax calculation.

Document:
{context}

Return JSON: {"vendor": "...", "total": ..., "tax": ..., "tax_correct": true/false}
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {field.value}")
    print()

    # ── 8. Prompt Template with Schema ──
    print("7. PROMPT TEMPLATE (your template + your schema)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_contract.pdf",
        model=MODEL,
        schema={
            "parties": {"type": "array", "description": "All parties involved"},
            "duration": {"type": "string", "description": "How long the agreement lasts"},
            "max_liability": {"type": "number", "description": "Maximum liability amount"},
        },
        prompt_template="""You are a legal document analyst.
Extract the requested fields from this contract.

Fields to extract:
{schema}

Contract text:
{context}

Return ONLY valid JSON with the field names as keys.
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        val = str(field.value)[:40]
        print(f"   {name:20s} = {val}")
    print()

    # ── 8. Structured Data Q&A ──
    print("9. STRUCTURED DATA Q&A (exact computation)")
    print("-" * 60)
    dq.ingest("demo/sample_sales.csv", embed=False)
    for q in ["What is the total amount?", "What is the average amount?"]:
        answer = dq.ask(q, source="demo/sample_sales.csv")
        print(f"   Q: {q}")
        print(f"   A: {answer}")
    print()

    # ── 9. Document Classification ──
    print("10. CLASSIFICATION")
    print("-" * 60)
    for doc_path in ["demo/sample_invoice.pdf", "demo/sample_contract.pdf"]:
        labels = dq.classify(doc_path)
        print(f"   {os.path.basename(doc_path):25s} -> {labels[0]['label']} ({labels[0]['confidence']:.0%})")

    shutil.rmtree(store, ignore_errors=True)
    print()
    print("=" * 60)
    print("DONE — All extractions powered by local Ollama LLM")
    print(f"Model used: {MODEL}")
    print("Your prompts. Your schema. Our pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
