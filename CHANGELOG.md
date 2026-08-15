# Changelog

## v0.3.1 (August 2026)

### RAG Strategy Selector
- Three RAG modes: GeneralRAG, GraphRAG, MultimodalRAG
- `dq.ask_rag("question", mode="graphrag")` — user chooses strategy
- Classification-first approach: classify document → pick RAG mode

### GraphRAG Engine
- Extract entities from documents → build knowledge graph → graph-enhanced retrieval
- Cross-document connections: Invoice → Vendor → Contract → Liability
- Evidence chain: every answer shows how it was derived through the graph
- `dq.build_document_graph()` — builds graph from all ingested documents

### Graph Visualization
- Interactive HTML visualization with vis.js
- Color-coded nodes: documents (green), organizations (orange), money (red), dates (cyan)
- Draggable, zoomable, hoverable nodes and edges
- `dq.visualize_graph(output="graph.html")` — generates standalone HTML

### New demo
- `demo/05_graphrag.py` — full GraphRAG pipeline with visualization

## v0.3.0 (August 2026)

### Source Attribution
- Every Q&A answer shows which document it came from
- `ask_with_source()` returns QAAnswer with .answer, .source_name, .confidence, .method

### FAISS Vector Store
- Production-scale vector search, scales to millions of documents
- `Docqwise(vector_store="faiss")`

### Q&A Engine Improvements
- Smart source routing for multi-document scenarios
- Direct file reading when no embedder available

## v0.2.0 (August 2026)

### Complete platform release
- Factory-based architecture, RAG extraction pipeline
- Custom prompts and prompt templates
- 10 readers, 36 formats, 4 extraction methods
- Self-improving corrections, incremental processing
- REST API, Docker, CLI

## v0.1.0

- Initial project scaffold
