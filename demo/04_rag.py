"""
DocQWise Demo 4: RAG-Based Extraction
=======================================

The core differentiator of DocQWise.
No truncation. Full document processed. Intelligent retrieval.

How it works:
  1. Read full document (any length, any format)
  2. Chunk with structure preservation (tables stay intact)
  3. Embed all chunks (semantic vectors)
  4. For each field: retrieve the most relevant chunks
  5. Send only relevant context to LLM
  6. Extract with full accuracy

Requirements:
    pip install docqwise
    pip install sentence-transformers

    # Plus one of these LLM backends:
    # Option A: Ollama (recommended)
    #   ollama pull nemotron-mini && ollama serve
    # Option B: HuggingFace
    #   pip install transformers torch bitsandbytes accelerate

Run:
    python demo/04_rag.py
"""

import os
import tempfile
import shutil
from docqwise import Docqwise
from docqwise.readers.pdf_reader import PDFReader
from docqwise.chunkers.structure_chunker import StructureChunker


def main():
    print("=" * 60)
    print("DocQWise — RAG-Based Document Extraction")
    print("No truncation. Full document. Intelligent retrieval.")
    print("=" * 60)
    print()

    store = os.path.join(tempfile.gettempdir(), "docqwise_rag_demo")
    if os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)

    dq = Docqwise(store_path=store)

    # ── Show the RAG Pipeline ──
    print("PIPELINE:")
    print("  Document → Chunk → Embed → Retrieve relevant → LLM Extract")
    print()

    # ── 1. Read Full Document ──
    print("1. READ FULL DOCUMENT")
    print("-" * 60)
    reader = PDFReader()
    doc = reader.read("demo/sample_invoice.pdf")
    print(f"   Source: {doc.source_path}")
    print(f"   Pages: {doc.metadata.page_count}")
    print(f"   Words: {doc.metadata.word_count}")
    print(f"   Full text length: {len(doc.text)} characters")
    print(f"   NO TRUNCATION — using entire document")
    print()

    # ── 2. Chunk with Structure Preservation ──
    print("2. STRUCTURE-PRESERVING CHUNKING")
    print("-" * 60)
    chunker = StructureChunker(max_tokens=300, overlap=50)
    chunks = chunker.chunk(doc)
    print(f"   Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        etype = chunk.metadata.element_type.value
        words = len(chunk.text.split())
        heading = chunk.metadata.heading_context or "(no heading)"
        print(f"   Chunk {i+1}: [{etype:6s}] {words:3d} words | {heading}")
    print()

    # ── 3. Embed Chunks ──
    print("3. EMBED CHUNKS")
    print("-" * 60)
    try:
        from docqwise.embedders.sentence_transformer import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        print(f"   Embedding model: all-MiniLM-L6-v2")
        print(f"   Dimension: {embedder.dimension()}")
        print(f"   Chunks embedded: {len(chunks)}")
    except ImportError:
        print("   sentence-transformers not installed — using positional retrieval")
        embedder = None
    print()

    # ── 4. Retrieve Relevant Chunks ──
    print("4. RETRIEVE RELEVANT CHUNKS")
    print("-" * 60)
    if embedder:
        import numpy as np
        queries = {
            "invoice_number": "invoice number ID reference",
            "total_amount": "total amount payment due",
            "vendor": "vendor company name seller from",
            "line_items": "items products services quantity price",
        }
        for field_name, query in queries.items():
            query_emb = embedder.embed(query)
            scored = []
            for chunk in chunks:
                if chunk.embedding is not None:
                    sim = float(np.dot(query_emb, chunk.embedding) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(chunk.embedding) + 1e-10
                    ))
                    scored.append((chunk, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[0] if scored else None
            if top:
                print(f"   Query '{field_name}':")
                print(f"     Best chunk (score={top[1]:.3f}): {top[0].text[:80]}...")
        print()

    # ── 5. RAG Extraction (Full Pipeline) ──
    print("5. RAG EXTRACTION — INVOICE")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        template="invoice",
        method="rag",
    )
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Confidence: {result.confidence:.2f}")
    for name, field in result.fields.items():
        page = f"page {field.page}" if field.page else ""
        print(f"   {name:20s} = {str(field.value):25s} [{field.confidence:.2f}] {page}")
    print()

    # ── 6. RAG Extraction — Contract ──
    print("6. RAG EXTRACTION — CONTRACT")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_contract.pdf",
        template="contract",
        method="rag",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        val = str(field.value)[:40]
        print(f"   {name:20s} = {val}")
    print()

    # ── 7. RAG with Custom Schema ──
    print("7. RAG — CUSTOM SCHEMA")
    print("-" * 60)
    schema = {
        "sender": {"type": "string", "description": "Who sent this document"},
        "receiver": {"type": "string", "description": "Who receives this document"},
        "total_value": {"type": "number", "description": "Total monetary value"},
        "all_dates": {"type": "array", "description": "Every date mentioned"},
        "purpose": {"type": "string", "description": "Purpose of this document"},
    }
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        schema=schema,
        method="rag",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value):40s}")
    print()

    # ── 8. RAG with Custom Prompt ──
    print("8. RAG — YOUR PROMPT (you design, pipeline retrieves context)")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_invoice.pdf",
        method="rag",
        prompt="""You are a financial auditor reviewing this invoice.
Identify any potential issues or missing information.
Extract: vendor, total, tax, and flag if tax calculation seems incorrect.

Document:
{context}

Return JSON: {"vendor": "...", "total": ..., "tax": ..., "tax_correct": true/false, "issues": ["..."]}
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        print(f"   {name:20s} = {str(field.value):40s}")
    print()

    # ── 9. RAG with Prompt Template + Schema ──
    print("9. RAG — YOUR TEMPLATE + YOUR SCHEMA")
    print("-" * 60)
    result = dq.extract_fields(
        "demo/sample_contract.pdf",
        method="rag",
        schema={
            "risk_level": {"type": "string", "description": "Low, Medium, or High risk"},
            "key_obligations": {"type": "array", "description": "List of obligations"},
            "missing_clauses": {"type": "array", "description": "Important clauses that are missing"},
        },
        prompt_template="""You are a legal risk analyst.
Analyze this contract for risks and completeness.

Extraction requirements:
{schema}

Contract sections:
{context}

Return JSON with the exact field names from the schema.
JSON:""",
    )
    print(f"   Strategy: {result.strategy_used}")
    for name, field in result.fields.items():
        val = str(field.value)[:50]
        print(f"   {name:20s} = {val}")
    print()

    # ── 10. Method Comparison ──
    print("10. METHOD COMPARISON")
    print("-" * 60)
    for method in ["rag", "llm", "regex"]:
        r = dq.extract_fields("demo/sample_invoice.pdf", template="invoice", method=method)
        print(f"    {method:6s} → strategy={r.strategy_used:6s} fields={len(r.fields):2d} confidence={r.confidence:.2f}")
    print()

    # ── 11. Structured Data Q&A ──
    print("11. STRUCTURED DATA Q&A")
    print("-" * 60)
    dq.ingest("demo/sample_sales.csv", embed=False)
    for q in ["What is the total amount?", "Which vendor has the highest sales?"]:
        answer = dq.ask(q, source="demo/sample_sales.csv")
        print(f"    Q: {q}")
        print(f"    A: {answer}")
    print()

   
    print("You design the prompts. We handle the pipeline.")
    print("=" * 60)

    shutil.rmtree(store, ignore_errors=True)


if __name__ == "__main__":
    main()
