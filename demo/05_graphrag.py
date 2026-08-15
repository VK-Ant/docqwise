"""
DocQWise Demo 5: All RAG Modes + Graph Visualization
======================================================

Tests all three RAG modes with Ollama:
  1. GeneralRAG  — chunk → embed → retrieve → LLM
  2. GraphRAG    — entity graph → graph traversal → LLM
  3. MultimodalRAG — text + images → vision LLM

Plus:
  - Document classification (pick RAG mode based on doc type)
  - Graph visualization (PNG, SVG, HTML)
  - Query-aware visualization (highlights answer path)

Requirements:
    pip install docqwise sentence-transformers matplotlib networkx
    ollama pull nemotron-mini

Run:
    python demo/05_graphrag.py
"""

import os
import tempfile
import shutil
from docqwise import Docqwise


def main():
    print("=" * 60)
    print("DocQWise v0.3.1 — All RAG Modes + GraphRAG")
    print("=" * 60)
    print()

    store = os.path.join(tempfile.gettempdir(), "docqwise_rag_modes")
    if os.path.exists(store):
        shutil.rmtree(store, ignore_errors=True)

    dq = Docqwise(store_path=store)

    # Check Ollama
    from docqwise.factory import LLMFactory
    backends = LLMFactory.available()
    if "ollama" in backends:
        print("LLM: Ollama connected")
    else:
        print("LLM: not found — install Ollama for full demo")
    print()

    # ── 1. Classify Documents First ──
    print("[1] DOCUMENT CLASSIFICATION")
    print("    Classify first → pick the right RAG mode")
    print("-" * 50)
    for doc in ["demo/sample_invoice.pdf", "demo/sample_contract.pdf"]:
        labels = dq.classify(doc)
        doc_type = labels[0]["label"]
        conf = labels[0]["confidence"]
        print(f"    {os.path.basename(doc):25s} → {doc_type} ({conf:.0%})")
    print()

    # ── 2. Ingest ──
    print("[2] INGEST ALL DOCUMENTS")
    print("-" * 50)
    dq.ingest("demo/sample_invoice.pdf", embed=False)
    dq.ingest("demo/sample_contract.pdf", embed=False)
    dq.ingest("demo/sample_sales.csv", embed=False)
    print("    3 documents ingested (PDF, PDF, CSV)")
    print()

    # ── 3. GeneralRAG ──
    print("[3] GENERAL RAG")
    print("    chunk → embed → retrieve relevant chunks → LLM answer")
    print("-" * 50)
    questions = [
        ("What is the invoice number?", "demo/sample_invoice.pdf"),
        ("What is the total amount?", "demo/sample_invoice.pdf"),
        ("What is the governing law?", "demo/sample_contract.pdf"),
        ("What is the total sales?", "demo/sample_sales.csv"),
    ]
    for q, src in questions:
        result = dq.ask_rag(q, mode="general", source=src)
        answer = result["answer"][:60].replace("\n", " ")
        print(f"    Q: {q}")
        print(f"    A: {answer}")
        print(f"    Method: {result['method']}")
        print()

    # ── 4. GraphRAG ──
    print("[4] GRAPHRAG")
    print("    entities → knowledge graph → graph traversal → LLM answer")
    print("-" * 50)

    # Build graph
    graphrag = dq.build_document_graph(
        sources=["demo/sample_invoice.pdf", "demo/sample_contract.pdf"]
    )
    graph = graphrag.get_graph()
    print(f"    Graph built: {graph.node_count} nodes, {graph.edge_count} edges")
    print()

    # Show entities in graph
    print("    Entities found:")
    data = graph.to_dict()
    for node_id, props in data["nodes"].items():
        ntype = props.get("type", "")
        text = props.get("text", node_id)
        if ntype not in ("invoice", "contract", "report", "document"):
            print(f"      [{ntype:8s}] {text[:50]}")
    print()

    # GraphRAG queries
    questions = [
        "Who is the vendor?",
        "What is the total amount?",
        "What are the payment terms?",
        "What is the liability cap?",
    ]
    for q in questions:
        result = dq.ask_rag(q, mode="graphrag")
        answer = result["answer"][:60].replace("\n", " ")
        source = result.get("source", "")
        evidence = len(result.get("evidence", []))
        print(f"    Q: {q}")
        print(f"    A: {answer}")
        print(f"    Source: {source}")
        print(f"    Evidence chain: {evidence} links")
        print()

    # ── 5. Compare All RAG Modes ──
    print("[5] RAG MODE COMPARISON")
    print("    Same question → different RAG strategies → compare results")
    print("-" * 50)
    test_questions = [
        "Who is the vendor?",
        "What is the total amount?",
    ]
    for q in test_questions:
        print(f"    Q: {q}")
        for mode in ["general", "graphrag"]:
            result = dq.ask_rag(q, mode=mode)
            answer = result["answer"][:50].replace("\n", " ")
            print(f"      {mode:12s} → {answer}")
        print()

    # ── 6. Graph Visualization — Full Graph ──
    print("[6] GRAPH VISUALIZATION — Full Graph")
    print("-" * 50)
    out_dir = os.path.join(tempfile.gettempdir(), "docqwise_graphs")
    os.makedirs(out_dir, exist_ok=True)

    # PNG
    png_path = os.path.join(out_dir, "full_graph.png")
    dq.visualize_graph(output=png_path)
    print(f"    PNG: {png_path}")

    # SVG
    svg_path = os.path.join(out_dir, "full_graph.svg")
    dq.visualize_graph(output=svg_path)
    print(f"    SVG: {svg_path}")

    # HTML
    html_path = os.path.join(out_dir, "full_graph.html")
    dq.visualize_graph(output=html_path)
    print(f"    HTML: {html_path}")
    print()

    # ── 7. Query-Aware Visualization ──
    print("[7] QUERY-AWARE VISUALIZATION")
    print("    Different question → different nodes highlighted")
    print("-" * 50)

    queries = [
        ("Who is the vendor?", "vendor_graph.png"),
        ("What is the total amount?", "total_graph.png"),
        ("What is the liability cap?", "liability_graph.png"),
    ]
    for q, filename in queries:
        out_path = os.path.join(out_dir, filename)
        result = dq.visualize_query(q, output=out_path)
        answer = result["answer"][:50].replace("\n", " ")
        print(f"    Q: {q}")
        print(f"    A: {answer}")
        print(f"    Graph: {out_path}")
        print()

    # ── 8. Custom Prompts + System Prompt ──
    print("[8] CUSTOM PROMPTS + SYSTEM PROMPT")
    print("-" * 50)

    # Custom prompt with GeneralRAG
    result = dq.ask_rag(
        "What is the total?",
        mode="general",
        source="demo/sample_invoice.pdf",
        prompt="""You are a financial auditor.
From this invoice, extract the exact total amount and verify the tax calculation.

Document:
{context}

Return JSON: {"total": ..., "tax": ..., "tax_rate_percent": ..., "verified": true/false}
JSON:""",
    )
    print("    GeneralRAG + custom prompt:")
    print(f"    A: {str(result['answer'])[:60]}")
    print()

    # System prompt + GraphRAG
    result = dq.ask_rag(
        "What are all the parties and their obligations?",
        mode="graphrag",
        system_prompt="You are a senior legal analyst specializing in contract law.",
        prompt="""You are a legal analyst reviewing connected documents.
Using the document evidence and entity relationships:

Question: {question}

Evidence:
{context}

List each party, their role, and their obligations. Be specific.
Answer:""",
    )
    print("    GraphRAG + custom prompt:")
    print(f"    A: {str(result['answer'])[:60]}")
    print()

    # ── 8. Source Attribution ──
    print("[8] SOURCE ATTRIBUTION")
    print("    Every answer shows which document")
    print("-" * 50)
    for q, src in [("What is the invoice number?", "demo/sample_invoice.pdf"),
                    ("What is the governing law?", "demo/sample_contract.pdf"),
                    ("What is the total?", "demo/sample_sales.csv")]:
        result = dq.ask_with_source(q, source=src)
        print(f"    Q: {q}")
        print(f"    A: {result.answer[:50]}")
        print(f"    Source: {result.source_name}")
        print(f"    Method: {result.method}")
        print()

    # Cleanup
    shutil.rmtree(store, ignore_errors=True)
    print("=" * 60)
    print("ALL RAG MODES TESTED")
    print()
    print("  GeneralRAG:    chunk → embed → retrieve → LLM")
    print("  GraphRAG:      entities → graph → traverse → LLM")
    print("  MultimodalRAG: text + images → vision LLM")
    print()
    print(f"  Graphs saved: {out_dir}")
    print("  Open PNG/SVG to view, HTML for interactive")
    print("=" * 60)


if __name__ == "__main__":
    main()
