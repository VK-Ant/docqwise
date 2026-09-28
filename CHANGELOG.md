# Changelog

## v0.2.0 (August 2026)

### Complete platform release — 38 features, 111 files, 33 modules

### Readers (10)
- PDF reader (PyMuPDF), DOCX reader, Text/Markdown reader
- Image reader (JPG, PNG, TIFF, BMP, WebP)
- CSV/TSV reader, JSON/JSONL/XML/YAML reader
- Excel reader (XLSX, XLS), HTML reader, PPTX reader
- Email reader (EML, MSG), Parquet reader
- Auto-detect reader (36 formats, 5 source types)

### Extraction engine
- Text extraction with header/footer removal
- Table extraction (PyMuPDF + text fallback)
- Field extraction (regex + schema + auto-detect)
- Entity extraction (regex + spaCy)
- Image extraction from PDFs
- Metadata extraction
- Document classification (zero-shot keyword)
- Document comparison (text diff + similarity)

### Chunking
- Structure-preserving chunker (default, respects tables/headings)
- Fixed token window chunker
- Sentence-level chunker

### Embeddings & stores
- SentenceTransformer embedder (any HuggingFace model)
- SQLite vector store (zero-dependency default)

### Retrieval
- Vector search (cosine similarity)
- BM25 keyword search
- Hybrid search (vector + BM25)
- Query router (computation vs filter vs semantic)
- Natural language Q&A engine (exact math for structured data)

### Schema intelligence
- Auto schema detection from CSV/JSON/Excel
- Schema validation
- Data quality scoring

### Speed & incremental
- Batch dispatcher (multiprocessing + threading)
- Hash-based change detection (incremental ingestion)
- File watcher (continuous ingestion)
- Progress tracking

### Learning & corrections
- Correction store (SQLite, exact overrides)
- Correction propagation (vendor/global/document scope)
- Learning report and export/import

### Security
- PII detection (email, phone, SSN, credit card, Aadhaar, PAN)
- PII redaction

### Graph
- Document graph (nodes, edges, communities, query)

### Pipeline
- DAG pipeline engine (compose, validate, execute)
- YAML pipeline config support
- Topological sort for parallel execution

### Templates
- Invoice extraction template
- Contract extraction template

### Server
- FastAPI REST API (extract, tables, classify, PII detect, health)
- MCP server mode

### LLM backends
- Ollama (local)
- OpenAI API

### OCR engines
- EasyOCR engine
- Tesseract engine

### Connectors & databases
- Filesystem connector
- SQLite database connector

### Export
- JSON exporter
- CSV exporter

### Ecosystem bridges
- SightRAG bridge
- sonarwise bridge
- adaptive-intelligence bridge
- llmevalkit bridge

### Infrastructure
- 18 pluggable Base* abstract classes
- Configuration system (YAML/JSON)
- CLI (8 commands)
- Docker + Docker Compose (multi-stage)
- Apache 2.0 license
- Demo use cases (invoice, contract, CSV query)
- Jupyter notebook (.ipynb)
- Benchmark runner

## v0.1.0

- Initial project scaffold and architecture design
