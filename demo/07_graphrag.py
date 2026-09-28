"""Demo 07: GraphRAG + Knowledge Graph Visualization (v0.4.0)

Entity graph traversal with evidence chains and interactive visualization.

Usage:
    python demo/07_graphrag.py

Requirements:
    pip install docqwise[graph]
    # For visualization: pip install matplotlib networkx
"""

from docqwise import Docqwise


def demo_graphrag_query():
    """GraphRAG — graph-enhanced retrieval and Q&A."""
    print("=" * 60)
    print("DEMO: GraphRAG Query")
    print("=" * 60)

    dq = Docqwise()

    # Build knowledge graph from documents
    print("\nBuilding knowledge graph...")
    engine = dq.build_graph(sources=["demo/sample_invoice.pdf"])

    stats = engine.graph.stats()
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
          f"{stats['documents']} documents")

    # Query using graph context
    result = engine.query("What is the total amount?")
    print(f"\nAnswer: {result['answer'][:200]}")
    print(f"Method: {result['method']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Evidence: {len(result['evidence'])} items")
    for ev in result["evidence"][:3]:
        print(f"  - {ev['entity']} ({ev['type']})")


def demo_rag_modes():
    """Three RAG modes: general, graphrag, multimodal."""
    print("\n" + "=" * 60)
    print("DEMO: RAG Strategy Modes")
    print("=" * 60)

    dq = Docqwise()

    modes = ["general", "graphrag", "multimodal"]
    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        try:
            result = dq.ask_rag(
                "What are the key amounts?",
                mode=mode,
                sources=["demo/sample_invoice.pdf"],
            )
            print(f"  Answer: {result['answer'][:100]}...")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Method: {result['method']}")
        except Exception as e:
            print(f"  [skip] {e}")


def demo_source_attribution():
    """Source attribution — every answer tracks where it came from."""
    print("\n" + "=" * 60)
    print("DEMO: Source Attribution")
    print("=" * 60)

    from docqwise.retrieval.rag_strategy import QAAnswer

    # QAAnswer tracks source info
    answer = QAAnswer(
        answer="The total amount is $15,750.00",
        source="invoices/inv_1234.pdf",
        confidence=0.92,
        method="graphrag",
        evidence=[
            {"entity": "$15,750.00", "type": "MONEY", "sources": ["inv_1234.pdf"]},
            {"entity": "Acme Corp", "type": "ORG", "sources": ["inv_1234.pdf"]},
        ],
    )

    print(f"\nAnswer: {answer.answer}")
    print(f"Source: {answer.source_name}")
    print(f"Confidence: {answer.confidence:.2f}")
    print(f"Method: {answer.method}")
    print(f"Evidence: {len(answer.evidence)} items")
    print(f"Repr: {repr(answer)}")


def demo_graph_visualization():
    """Knowledge graph visualization (requires matplotlib + networkx)."""
    print("\n" + "=" * 60)
    print("DEMO: Graph Visualization")
    print("=" * 60)

    try:
        import matplotlib
        import networkx
    except ImportError:
        print("  [skip] Install: pip install matplotlib networkx")
        return

    dq = Docqwise()

    # Build graph
    engine = dq.build_graph(sources=["demo/sample_invoice.pdf"])

    if engine.graph.stats()["nodes"] == 0:
        print("  [skip] No entities extracted (needs LLM or regex matches)")
        return

    from docqwise.retrieval.graphrag import GraphVisualizer

    # Static PNG
    viz = GraphVisualizer(engine.graph)
    output = viz.to_image("demo/output_graph.png")
    print(f"\nStatic graph: {output}")

    # Interactive HTML
    output = viz.to_html("demo/output_graph.html")
    print(f"Interactive graph: {output}")

    # Query-aware visualization
    result = engine.query("What is the total?")
    viz = GraphVisualizer.from_query(engine.graph, result)
    output = viz.save("demo/output_query_graph.png")
    print(f"Query graph: {output}")


if __name__ == "__main__":
    demo_graphrag_query()
    demo_rag_modes()
    demo_source_attribution()
    demo_graph_visualization()
    print("\n" + "=" * 60)
    print("All GraphRAG demos complete!")
    print("=" * 60)
