"""
DocQWise Demo 3: HuggingFace (Local GPU)
==========================================

AI-powered extraction using HuggingFace models on your GPU.
Downloads model on first run, loads from cache after.

Requirements:
    pip install docqwise
    pip install transformers torch bitsandbytes accelerate sentencepiece

    # For 8GB VRAM: use 3B model (default)
    # For 16GB+ VRAM: change to 7B model in code

Run:
    python demo/03_huggingface.py
"""

import os
import tempfile
import shutil
from docqwise import Docqwise
from docqwise.llm.hf_llm import HuggingFaceLLM


def main():
    print("=" * 60)
    print("DocQWise + HuggingFace — GPU Document Extraction")
    print("=" * 60)
    print()

    store = os.path.join(tempfile.gettempdir(), "docqwise_hf_demo")
    if os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)

    # Initialize HuggingFace LLM
    # Choose your model:
    #   "Qwen/Qwen2.5-7B-Instruct"         — best accuracy (~4GB VRAM with 4bit)
    #   "Qwen/Qwen2.5-3B-Instruct"         — faster (~2GB VRAM with 4bit)
    #   "microsoft/Phi-3-mini-4k-instruct"  — lightest (~2GB VRAM with 4bit)
    #   "mistralai/Mistral-7B-Instruct-v0.3" — good balance (~4GB VRAM with 4bit)

    MODEL = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading model: {MODEL}")
    print("First run downloads the model. Subsequent runs load from cache.")
    print()

    llm = HuggingFaceLLM(
        model_name=MODEL,
        quantize="4bit",
    )

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
        llm=llm,
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
        llm=llm,
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        val = str(field.value)[:40]
        print(f"   {name:20s} = {val}")
    print()

    # ── 3. Resume Extraction ──
    print("3. RESUME EXTRACTION (template)")
    print("-" * 60)
    # Using invoice PDF as placeholder — replace with a real resume PDF
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        template="resume",
        llm=llm,
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value)[:35]}")
    print()

    # ── 4. Custom Schema ──
    print("4. CUSTOM SCHEMA EXTRACTION")
    print("-" * 60)
    schema = {
        "sender_company": {"type": "string", "description": "Company that sent this document"},
        "receiver_company": {"type": "string", "description": "Company receiving this document"},
        "total_value": {"type": "number", "description": "Total monetary value"},
        "key_dates": {"type": "array", "description": "All important dates"},
        "payment_method": {"type": "string", "description": "How payment should be made"},
    }
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        schema=schema,
        llm=llm,
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value):40s}")
    print()

    # ── 5. Custom Prompt — user designs the prompt ──
    print("5. CUSTOM PROMPT (you design, we run the pipeline)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        llm=llm,
        prompt="""You are a financial document analyzer.
From this invoice, extract:
1. The sender company
2. The receiver company
3. Total amount as a number
4. Whether GST/tax is included

Document:
{context}

Return JSON: {"sender": "...", "receiver": "...", "total": ..., "tax_included": true/false}
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {field.value}")
    print()

    # ── 6. System Prompt (set LLM role) ──
    print("6. SYSTEM PROMPT")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        llm=llm,
        system_prompt="You are an accounts payable specialist. Extract financial details precisely.",
        prompt="Extract vendor, total, tax from this invoice.

Document:
{context}

JSON:",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {field.value}")
    print()

    # ── 7. Prompt Template with Schema ──
    print("6. PROMPT TEMPLATE (your template + your schema)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_contract.pdf",
        llm=llm,
        schema={
            "parties": {"type": "array", "description": "All parties involved"},
            "duration": {"type": "string", "description": "Agreement duration"},
            "max_liability": {"type": "number", "description": "Maximum liability"},
            "governing_law": {"type": "string", "description": "Jurisdiction"},
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

    # ── 7. Auto Extract ──
    print("8. AUTO EXTRACT (LLM decides what to extract)")
    print("-" * 60)
    result = dq.auto_extract("demo/sample_invoice.pdf")
    print(f"   Strategy: {result.strategy_used}")
    for name, field in list(result.fields.items())[:8]:
        print(f"   {name:20s} = {str(field.value):30s}")
    print()

    # ── 8. Entity Extraction ──
    print("9. ENTITY EXTRACTION")
    print("-" * 60)
    entities = dq.extract_entities("demo/sample_contract.pdf")
    for e in entities[:8]:
        print(f"   [{e.entity_type:10s}] {e.text}")
    print()

    # ── 9. Structured Data Q&A ──
    print("10. STRUCTURED DATA Q&A")
    print("-" * 60)
    dq.ingest("demo/sample_sales.csv", embed=False)
    for q in ["What is the total amount?", "What is the average amount?"]:
        print(f"   Q: {q}")
        print(f"   A: {dq.ask(q, source='demo/sample_sales.csv')}")
    print()

    # ── 10. Free GPU Memory ──
    print("11. CLEANUP")
    print("-" * 60)
    llm.unload()
    print("    GPU memory freed")

    shutil.rmtree(store, ignore_errors=True)
    print()
    print("=" * 60)
    print("DONE — All extractions powered by HuggingFace on your GPU")
    print(f"Model: {MODEL} (4-bit quantized)")
    print("Your prompts. Your schema. Our pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
