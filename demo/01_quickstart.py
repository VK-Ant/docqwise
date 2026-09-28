"""
DocQWise Demo 1: Quick Start
==============================

AI-powered document intelligence. Works with Ollama locally.

This demo shows:
  - Read PDF, CSV, JSON documents
  - AI-powered field extraction (Ollama / HuggingFace)
  - Agentic multi-pass extraction
  - Table and entity extraction
  - Classify documents
  - Compare documents
  - Schema detection
  - Structured data Q&A
  - PII detection
  - Self-improving corrections

Requirements:
    pip install docqwise
    ollama pull nemotron-mini

Run:
    python demo/01_quickstart.py
"""

import os
import tempfile
import shutil
from docqwise import Docqwise


def main():
    print("=" * 60)
    print("DocQWise v0.4.0 — Quick Start Demo")
    print("Read. Extract. Retrieve.")
    print("=" * 60)
    print()

    # Setup
    store = os.path.join(tempfile.gettempdir(), "docqwise_quickstart")
    if os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)
    dq = Docqwise(store_path=store)

    # ── 1. Read and Ingest ──
    print("[1] INGEST DOCUMENTS")
    print("    Read any format — PDF, CSV, JSON, DOCX, Excel, images")
    print()
    for source in ["demo/sample_invoice.pdf", "demo/sample_contract.pdf",
                    "demo/sample_sales.csv", "demo/sample_data.json"]:
        r = dq.ingest(source, embed=False)
        name = os.path.basename(source)
        print(f"    {name:25s} ✓ processed in {r['elapsed_s']}s")
    print()

    # ── 2. Incremental ──
    print("[2] INCREMENTAL PROCESSING")
    print("    Re-ingest skips unchanged files automatically")
    print()
    r = dq.ingest("demo/sample_invoice.pdf", embed=False)
    print(f"    Re-ingest invoice: {r['processed']} processed (0 = skipped)")
    print()

    # ── 3. AI-Powered Field Extraction ──
    print("[3] AI-POWERED FIELD EXTRACTION")
    print("    Extract structured data using LLM (Ollama/HuggingFace)")
    print()
    result = dq.extract_fields("demo/sample_invoice.pdf", template="invoice")
    for name, field in result.fields.items():
        print(f"    {name:20s} = {str(field.value):25s} [{field.confidence:.2f}]")
    print()

    # ── 4. Agentic Extraction ──
    print("[4] AGENTIC EXTRACTION")
    print("    Multi-pass, self-correcting AI extraction")
    print()
    result = dq.extract_agentic("demo/sample_invoice.pdf", template="invoice")
    for name, field in result.fields.items():
        print(f"    {name:20s} = {str(field.value):25s} [{field.confidence:.2f}]")
    print(f"    Strategy: {result.strategy_used}")
    print()

    # ── 5. Custom Schema ──
    print("[5] CUSTOM SCHEMA")
    print("    Define exactly what fields you want")
    print()
    schema = {
        "sender": {"type": "string", "description": "Who sent this document"},
        "total_value": {"type": "number", "description": "Total monetary value"},
        "due_date": {"type": "date", "description": "Payment due date"},
    }
    result = dq.extract_fields("demo/sample_invoice.pdf", schema=schema)
    for name, field in result.fields.items():
        print(f"    {name:20s} = {field.value}")
    print()

    # ── 6. Extract Tables ──
    print("[6] TABLE EXTRACTION")
    tables = dq.extract_tables("demo/sample_invoice.pdf")
    print(f"    Tables found: {len(tables)}")
    print()

    # ── 7. Entity Extraction ──
    print("[7] ENTITY EXTRACTION")
    entities = dq.extract_entities("demo/sample_invoice.pdf")
    for e in entities[:5]:
        print(f"    [{e.entity_type:10s}] {e.text}")
    print()

    # ── 8. Classification ──
    print("[8] DOCUMENT CLASSIFICATION")
    for doc_path in ["demo/sample_invoice.pdf", "demo/sample_contract.pdf"]:
        labels = dq.classify(doc_path)
        top = labels[0]
        print(f"    {os.path.basename(doc_path):25s} → {top['label']} ({top['confidence']:.0%})")
    print()

    # ── 9. Comparison ──
    print("[9] DOCUMENT COMPARISON")
    diff = dq.compare("demo/sample_invoice.pdf", "demo/sample_contract.pdf")
    print(f"    Similarity: {diff['similarity']:.1%}")
    print(f"    Changes: +{diff['additions_count']} / -{diff['deletions_count']}")
    print()

    # ── 10. Schema Detection ──
    print("[10] SCHEMA DETECTION")
    schema = dq.detect_schema("demo/sample_sales.csv")
    for f in schema["fields"][:5]:
        print(f"    {f['name']:15s} → {f['type']}")
    print()

    # ── 11. Q&A on Structured Data ──
    print("[11] STRUCTURED DATA Q&A")
    print("     Exact computation — real math, not LLM guessing")
    print()
    for q in ["What is the total amount?", "What is the average amount?",
              "How many items are there?"]:
        a = dq.ask(q, source="demo/sample_sales.csv")
        print(f"     Q: {q}")
        print(f"     A: {a}")
    print()

    # ── 11. PII Detection ──
    print("[11] PII DETECTION")
    pii = dq.detect_pii("demo/sample_invoice.pdf")
    for p in pii:
        print(f"     [{p.pii_type:12s}] {p.value}")
    print()

    # ── 12. Self-Improving Corrections ──
    print("[12] SELF-IMPROVING CORRECTIONS")
    print("     Fix once → auto-applied on future similar documents")
    print()
    result = dq.extract_fields("demo/sample_invoice.pdf", template="invoice")
    result.correct({"tax": 33300.00, "gst_number": "29AABCU9603R1ZM"})
    print(f"     Corrected: tax=33300.0, gst_number=29AABCU9603R1ZM")
    print(f"     Stored: {dq.learning_report()['total_corrections']} corrections")
    print()

    # ── 13. Pipeline + Graph ──
    print("[13] PIPELINE DAG")
    from docqwise.pipeline.pipeline import Pipeline
    pipe = Pipeline("extraction_pipeline")
    pipe.add_node("read", node_type="reader")
    pipe.add_node("extract", node_type="extractor")
    pipe.add_node("store", node_type="store")
    pipe.connect("read", "extract")
    pipe.connect("extract", "store")
    print(f"     Valid: {pipe.validate()}")
    print(f"     {pipe.visualize()}")
    print()

    print("[14] DOCUMENT GRAPH")
    from docqwise.graph.document_graph import DocumentGraph
    graph = DocumentGraph()
    graph.add_document("invoice", {"type": "invoice"})
    graph.add_document("contract", {"type": "contract"})
    graph.add_entity("acme", "ORG", "Acme Corporation")
    graph.add_edge("invoice", "acme", "from_vendor")
    graph.add_edge("contract", "acme", "with_party")
    print(f"     Nodes: {graph.node_count}, Edges: {graph.edge_count}")
    print(f"     Communities: {graph.communities()}")
    print()

    # Cleanup
    shutil.rmtree(store, ignore_errors=True)
    print("=" * 60)
    print("ALL 15 FEATURES WORKING")
    print()
    print("Next: Try AI-powered extraction →")
    print("  python demo/02_ollama.py      (local Ollama LLM)")
    print("  python demo/03_huggingface.py (local HuggingFace model)")
    print("  python demo/04_rag.py         (RAG-based extraction)")
    print("=" * 60)


if __name__ == "__main__":
    main()
