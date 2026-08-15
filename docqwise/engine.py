"""DocQWise Engine — stable core. Never changes when adding features.

All component selection goes through factories.
Adding a new extractor, LLM, store = register in factory.py, not here.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional, Union

from docqwise._version import __version__
from docqwise.config import DocqwiseConfig
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk
from docqwise.core.field import ExtractionResult
from docqwise.core.element import Table, ExtractedImage, Entity, Relation

logger = logging.getLogger("docqwise")


class Docqwise:
    """DocQWise: Read. Extract. Retrieve.

    Usage:
        from docqwise import Docqwise

        dq = Docqwise()
        dq.ingest("documents/")
        result = dq.extract_fields("invoice.pdf", template="invoice")
        answer = dq.ask("What is the total amount?")
    """

    def __init__(
        self,
        strategy: str = "auto",
        workers: Union[int, str] = 1,
        gpu_workers: int = 0,
        threads: int = 4,
        batch_size: int = 32,
        ocr: Any = "easyocr",
        embedder: Any = None,
        llm: Any = None,
        vector_store: Any = "sqlite",
        store_path: str = "./docqwise_db",
        search_mode: str = "vector",
        incremental: bool = True,
        cache: bool = True,
        config: Optional[Union[str, DocqwiseConfig]] = None,
    ):
        # Load config
        if isinstance(config, str):
            self._config = DocqwiseConfig.from_yaml(config)
        elif isinstance(config, DocqwiseConfig):
            self._config = config
        else:
            self._config = DocqwiseConfig()
            self._config.strategy.mode = strategy
            self._config.speed.workers = workers
            self._config.speed.gpu_workers = gpu_workers
            self._config.speed.threads = threads
            self._config.speed.batch_size = batch_size
            self._config.storage.vector_store = (
                vector_store if isinstance(vector_store, str) else "custom"
            )
            self._config.storage.store_path = store_path
            self._config.storage.cache = cache
            self._config.search.mode = search_mode
            self._config.incremental.enabled = incremental

        # Store user-provided components
        self._user_llm = llm
        self._user_embedder = embedder
        self._user_vector_store = vector_store if not isinstance(vector_store, str) else None
        self._store_path = store_path

        # Internal components — initialized lazily
        self._reader = None
        self._vector_store = None
        self._correction_store = None
        self._change_detector = None
        self._qa_engine = None
        self._initialized = False

    # ══════════════════════════════════════
    # INITIALIZATION (lazy)
    # ══════════════════════════════════════

    def _ensure_initialized(self):
        if self._initialized:
            return

        from docqwise.readers.auto_reader import AutoReader
        from docqwise.factory import StoreFactory
        from docqwise.learning.correction import CorrectionStore
        from docqwise.incremental.change_detector import ChangeDetector
        from docqwise.retrieval.qa_engine import QAEngine

        self._reader = AutoReader()
        self._vector_store = StoreFactory.get(
            store=self._user_vector_store,
            store_type=self._config.storage.vector_store,
            store_path=self._store_path,
        )
        self._correction_store = CorrectionStore(self._store_path)
        self._change_detector = ChangeDetector(self._store_path)
        self._qa_engine = QAEngine(self._vector_store)
        self._initialized = True

    # ══════════════════════════════════════
    # INGESTION
    # ══════════════════════════════════════

    def ingest(self, source: str, **kwargs) -> dict:
        """Ingest documents from file, folder, URL, or database."""
        self._ensure_initialized()
        start = time.time()

        from docqwise.connectors.filesystem import FilesystemConnector
        from docqwise.factory import ChunkerFactory, EmbedderFactory

        fs = FilesystemConnector()

        # Resolve source to file list
        if os.path.isfile(source):
            files = [source]
        elif os.path.isdir(source):
            files = fs.list_files(source, extensions=self._reader.supported_formats())
        elif "://" in source:
            return self._ingest_structured(source, **kwargs)
        else:
            return {"error": f"Source not found: {source}"}

        # Incremental: skip unchanged
        if self._config.incremental.enabled:
            files = [f for f in files if self._change_detector.has_changed(f)]

        chunker = ChunkerFactory.get(
            strategy=self._config.chunker.strategy,
            max_tokens=self._config.chunker.max_chunk_tokens,
            overlap=self._config.chunker.overlap_tokens,
        )

        # Embed if requested
        embedder = None
        if kwargs.get("embed", True):
            embedder = EmbedderFactory.get(self._user_embedder)

        processed, errors = 0, []
        for fpath in files:
            try:
                doc = self._reader.read(fpath)
                chunks = chunker.chunk(doc)

                # Register structured data for Q&A
                if doc.metadata.custom.get("rows"):
                    self._qa_engine.register_structured(
                        fpath, doc.metadata.custom.get("headers", []),
                        doc.metadata.custom["rows"],
                    )

                # Embed chunks
                if embedder and chunks:
                    try:
                        texts = [c.text for c in chunks]
                        embeddings = embedder.embed_batch(texts)
                        for chunk, emb in zip(chunks, embeddings):
                            chunk.embedding = emb
                            chunk.metadata.source_path = fpath
                    except Exception:
                        pass

                # Store
                if self._vector_store and chunks:
                    self._vector_store.insert(chunks)

                # Mark processed
                if self._config.incremental.enabled:
                    self._change_detector.mark_processed(fpath)
                processed += 1
            except Exception as e:
                errors.append({"file": fpath, "error": str(e)})

        elapsed = time.time() - start
        return {
            "processed": processed, "errors": len(errors),
            "total_files": len(files), "elapsed_s": round(elapsed, 2),
            "error_details": errors,
        }

    def _ingest_structured(self, source: str, **kwargs) -> dict:
        tables = kwargs.get("tables", None)
        if source.startswith("sqlite:///"):
            from docqwise.databases.sqlite_db import SQLiteDB
            db = SQLiteDB()
            db.connect(source)
            table_names = tables or db.tables()
            for tbl in table_names:
                rows = db.query(f"SELECT * FROM {tbl}")
                if rows:
                    self._qa_engine.register_structured(
                        f"{source}/{tbl}", list(rows[0].keys()), rows,
                    )
            return {"processed": len(table_names), "source": source}
        return {"error": f"Database type not yet supported: {source}"}

    async def ingest_async(self, source: str, **kwargs) -> dict:
        return self.ingest(source, **kwargs)

    def sync(self, source: str = None, mode: str = "incremental",
             interval: int = 300) -> None:
        raise NotImplementedError("Database sync coming soon")

    def watch(self, path: str, poll_interval: int = 60, on_new: Any = None) -> None:
        from docqwise.incremental.watcher import FileWatcher
        watcher = FileWatcher(path, callback=on_new or (lambda f: self.ingest(f)),
                              interval=poll_interval)
        watcher.start()

    # ══════════════════════════════════════
    # EXTRACTION (all through factory)
    # ══════════════════════════════════════

    def extract_fields(self, source: str, schema: dict = None,
                       template: str = None, method: str = "auto",
                       llm=None, prompt: str = None,
                       prompt_template: str = None,
                       system_prompt: str = None, **kwargs) -> ExtractionResult:
        """Extract fields from a document.

        Args:
            source: file path
            schema: custom field definitions
            template: pre-built template (invoice, contract, resume, receipt)
            method: auto | rag | llm | vision | regex
            llm: your own LLM instance
            prompt: your extraction prompt — use {context} for document text
            prompt_template: your template — use {context} and {schema} placeholders
            system_prompt: LLM system role (e.g. "You are a maritime document specialist")
            model: Ollama model name
        """
        self._ensure_initialized()
        doc = self._reader.read(source)

        # Resolve template schema
        target_schema = schema
        if template:
            from docqwise.factory import TemplateFactory
            tmpl = TemplateFactory.get(template)
            if tmpl:
                target_schema = tmpl.SCHEMA

        # Get extractor from factory
        from docqwise.factory import ExtractorFactory
        model = kwargs.pop("model", None)
        extractor = ExtractorFactory.get(
            method=method,
            llm=llm or self._user_llm,
            embedder=self._user_embedder,
            model=model,
            **kwargs,
        )

        # For vision extractor, different call signature
        if method == "vision":
            result = extractor.extract_from_pdf_page(source) if source.endswith(".pdf") \
                else extractor.extract_from_image(source, schema=target_schema)
        else:
            result = extractor.extract_fields(
                doc, schema=target_schema,
                prompt=prompt, prompt_template=prompt_template,
                system_prompt=system_prompt,
            )

        result._engine = self
        return self._apply_corrections(result, doc)

    def extract_fields_batch(self, source: str, **kwargs) -> list[ExtractionResult]:
        self._ensure_initialized()
        from docqwise.connectors.filesystem import FilesystemConnector
        files = FilesystemConnector().list_files(
            source, extensions=self._reader.supported_formats(),
        )
        return [self.extract_fields(f, **kwargs) for f in files]

    def extract_all(self, source: str, **kwargs) -> DocqwiseDocument:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.extractors.table_extractor import RuleBasedTableExtractor
        from docqwise.extractors.image_extractor import ImageExtractor
        doc.tables = RuleBasedTableExtractor().extract_tables(doc)
        doc.images = ImageExtractor().extract_images(doc)
        doc.entities = self.extract_entities(source, **kwargs)
        return doc

    def extract_text(self, source: str, **kwargs) -> str:
        self._ensure_initialized()
        return self._reader.read(source).text

    def extract_tables(self, source: str, **kwargs) -> list[Table]:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.extractors.table_extractor import RuleBasedTableExtractor
        return RuleBasedTableExtractor().extract_tables(doc, **kwargs)

    def extract_images(self, source: str, **kwargs) -> list[ExtractedImage]:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.extractors.image_extractor import ImageExtractor
        return ImageExtractor().extract_images(doc, **kwargs)

    def extract_entities(self, source: str, types=None,
                         custom_types=None, method: str = "auto",
                         llm=None, **kwargs) -> list[Entity]:
        self._ensure_initialized()
        doc = self._reader.read(source)
        if method == "regex":
            from docqwise.extractors.entity_extractor import RegexEntityExtractor
            return RegexEntityExtractor().extract_entities(doc, types=types)
        from docqwise.extractors.llm_extractor import LLMEntityExtractor
        return LLMEntityExtractor(llm=llm or self._user_llm).extract_entities(
            doc, types=types, custom_types=custom_types,
        )

    def extract_form(self, source: str) -> dict:
        self._ensure_initialized()
        return {}

    def extract_layout(self, source: str) -> dict:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.strategy.profiler import DocumentProfiler
        return DocumentProfiler().profile(doc)

    def auto_extract(self, source: str, llm=None, **kwargs) -> ExtractionResult:
        """Auto-detect document type and extract using RAG."""
        self._ensure_initialized()
        doc = self._reader.read(source)

        from docqwise.classifier.zero_shot import ZeroShotClassifier
        from docqwise.factory import TemplateFactory, ExtractorFactory

        labels = ZeroShotClassifier().classify(doc.text)
        top_label = labels[0]["label"] if labels else None
        tmpl = TemplateFactory.get(top_label) if top_label else None
        target_schema = tmpl.SCHEMA if tmpl else None

        extractor = ExtractorFactory.get(
            method="auto", llm=llm or self._user_llm,
            model=kwargs.get("model"),
        )
        result = extractor.extract_fields(doc, schema=target_schema)
        result._engine = self
        return result

    # ══════════════════════════════════════
    # RETRIEVAL + Q&A
    # ══════════════════════════════════════

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        self._ensure_initialized()
        from docqwise.factory import EmbedderFactory
        embedder = EmbedderFactory.get(self._user_embedder)
        if embedder and self._vector_store:
            return self._vector_store.search(embedder.embed(query), top_k=top_k,
                                              filters=kwargs.get("filters"))
        return []

    def _ensure_qa_embedder(self):
        if self._qa_engine._embedder is None:
            from docqwise.factory import EmbedderFactory
            embedder = EmbedderFactory.get(self._user_embedder)
            if embedder:
                self._qa_engine._embedder = embedder

    def ask(self, question: str, source: str = None, **kwargs) -> str:
        self._ensure_initialized()
        self._ensure_qa_embedder()
        return self._qa_engine.ask(question, source=source)

    def ask_with_source(self, question: str, source: str = None, **kwargs):
        """Ask a question and get answer with source document attribution.

        Returns:
            QAAnswer with .answer, .source, .source_name, .confidence, .method
        """
        self._ensure_initialized()
        self._ensure_qa_embedder()
        return self._qa_engine.ask_with_source(question, source=source)

    def ask_rag(self, question: str, mode: str = "general",
                source: str = None, prompt: str = None,
                system_prompt: str = None, **kwargs) -> dict:
        """Ask using specific RAG strategy.

        Modes:
            general     — chunk → embed → retrieve → LLM (default)
            graphrag    — extract entities → build graph → graph retrieval → LLM
            multimodal  — text + images → vision LLM → combined

        Args:
            question: your question
            mode: general | graphrag | multimodal
            source: specific document path (optional)
            prompt: custom LLM prompt — use {context} for document text, {question} for the question
        """
        self._ensure_initialized()
        self._ensure_qa_embedder()
        from docqwise.retrieval.rag_strategy import RAGStrategy
        strategy = RAGStrategy(engine=self, mode=mode)
        return strategy.ask(question, source=source, prompt=prompt, **kwargs)

    def build_document_graph(self, sources: list[str] = None) -> Any:
        """Build knowledge graph from documents for GraphRAG."""
        self._ensure_initialized()
        from docqwise.retrieval.graphrag import GraphRAGEngine
        graphrag = GraphRAGEngine(self)
        graphrag.build_from_ingested(sources=sources)
        self._doc_graph = graphrag
        return graphrag

    def visualize_graph(self, output: str = "docqwise_graph.png",
                        sources: list[str] = None) -> str:
        """Generate graph visualization as image or HTML.

        Args:
            output: file path — .png, .svg, .pdf, .jpg, or .html
            sources: list of document paths (builds from these, or uses all ingested)

        Returns:
            path to saved file
        """
        self._ensure_initialized()
        if not hasattr(self, '_doc_graph') or self._doc_graph is None:
            self.build_document_graph(sources=sources)

        from docqwise.retrieval.graphrag import GraphVisualizer
        graph = self._doc_graph.get_graph()
        return GraphVisualizer.save(graph, output=output)

    def visualize_query(self, question: str, output: str = "docqwise_query_graph.png",
                        sources: list[str] = None) -> dict:
        """Ask a question with GraphRAG and visualize the answer path.

        Returns:
            dict with "answer", "source", "evidence", "graph_file"
        """
        self._ensure_initialized()
        if not hasattr(self, '_doc_graph') or self._doc_graph is None:
            self.build_document_graph(sources=sources)

        result = self._doc_graph.query(question)

        from docqwise.retrieval.graphrag import GraphVisualizer
        graph = self._doc_graph.get_graph()
        GraphVisualizer.from_query(graph, result, output=output)

        result["graph_file"] = output
        return result

    def sql(self, query: str) -> list[dict]:
        from docqwise.databases.sqlite_db import SQLiteDB
        db = SQLiteDB(os.path.join(self._store_path, "structured.db"))
        return db.query(query)

    # ══════════════════════════════════════
    # CLASSIFICATION + COMPARISON
    # ══════════════════════════════════════

    def classify(self, source: str, **kwargs) -> list[dict]:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.classifier.zero_shot import ZeroShotClassifier
        return ZeroShotClassifier().classify(doc.text, labels=kwargs.get("labels"))

    def classify_batch(self, source: str, **kwargs) -> list:
        self._ensure_initialized()
        from docqwise.connectors.filesystem import FilesystemConnector
        files = FilesystemConnector().list_files(
            source, extensions=self._reader.supported_formats(),
        )
        return [{"file": f, "labels": self.classify(f, **kwargs)} for f in files]

    def compare(self, source_a: str, source_b: str) -> dict:
        self._ensure_initialized()
        doc_a = self._reader.read(source_a)
        doc_b = self._reader.read(source_b)
        from docqwise.comparator.text_diff import TextComparator
        return TextComparator().compare(doc_a.text, doc_b.text)

    def compare_batch(self, pairs: list, **kwargs) -> list:
        return [self.compare(a, b) for a, b in pairs]

    def find_duplicates(self, source: str, threshold: float = 0.85) -> list:
        raise NotImplementedError("Deduplication coming soon")

    # ══════════════════════════════════════
    # SCHEMA
    # ══════════════════════════════════════

    def detect_schema(self, source: str) -> dict:
        self._ensure_initialized()
        doc = self._reader.read(source)
        rows = doc.metadata.custom.get("rows", [])
        headers = doc.metadata.custom.get("headers", [])
        if not rows or not headers:
            return {"error": "No structured data found"}
        from docqwise.schema.detector import SchemaDetector
        return SchemaDetector().detect(headers, rows)

    def validate(self, source: str, schema: dict) -> dict:
        self._ensure_initialized()
        doc = self._reader.read(source)
        rows = doc.metadata.custom.get("rows", [])
        from docqwise.schema.validator import SchemaValidator
        return SchemaValidator().validate(rows, schema)

    # ══════════════════════════════════════
    # GRAPH
    # ══════════════════════════════════════

    def build_graph(self, source: str = None) -> Any:
        from docqwise.graph.document_graph import DocumentGraph
        return DocumentGraph()

    # ══════════════════════════════════════
    # SECURITY
    # ══════════════════════════════════════

    def detect_pii(self, source: str) -> list:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.security.pii_detector import PIIDetector
        return PIIDetector().detect(doc.text)

    def redact(self, source: str, output: str, redact_types: list = None) -> None:
        self._ensure_initialized()
        doc = self._reader.read(source)
        from docqwise.security.pii_detector import PIIDetector
        redacted = PIIDetector().redact(doc.text, types=redact_types)
        with open(output, "w") as f:
            f.write(redacted)

    # ══════════════════════════════════════
    # LEARNING + CORRECTIONS
    # ══════════════════════════════════════

    def _apply_corrections(self, result: ExtractionResult,
                            doc: DocqwiseDocument) -> ExtractionResult:
        if self._correction_store and self._correction_store.count() > 0:
            corrected = self._correction_store.apply(
                doc.doc_type.value, result.to_dict(),
            )
            from docqwise.core.field import FieldResult
            for k, v in corrected.items():
                if k in result.fields and result.fields[k].value != v:
                    result.fields[k] = FieldResult(
                        value=v, confidence=1.0,
                        extraction_method="correction", is_corrected=True,
                    )
        return result

    def learning_report(self) -> dict:
        self._ensure_initialized()
        return {
            "total_corrections": self._correction_store.count(),
            "corrections": self._correction_store.list_corrections(),
        }

    def export_learnings(self, path: str):
        self._ensure_initialized()
        import json
        with open(path, "w") as f:
            json.dump(self._correction_store.list_corrections(), f, indent=2, default=str)

    def import_learnings(self, path: str):
        self._ensure_initialized()
        import json
        with open(path) as f:
            for c in json.load(f):
                self._correction_store.save(
                    c.get("doc_id", ""), c.get("source_path", ""),
                    {c["field_name"]: json.loads(c["correct_value"])},
                )

    # ══════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════

    def export(self, path: str, format: str = "json", **kwargs) -> None:
        if format == "json":
            from docqwise.export.json_exporter import JSONExporter
            JSONExporter().export([], path, **kwargs)
        elif format == "csv":
            from docqwise.export.csv_exporter import CSVExporter
            CSVExporter().export([], path, **kwargs)

    def to_dataframe(self, source: str, **kwargs):
        self._ensure_initialized()
        doc = self._reader.read(source)
        rows = doc.metadata.custom.get("rows", [])
        if rows:
            import pandas as pd
            return pd.DataFrame(rows)
        return None

    # ══════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════

    def metrics(self) -> dict:
        return {
            "version": __version__,
            "store_path": self._config.storage.store_path,
            "strategy": self._config.strategy.mode,
            "workers": self._config.speed.workers,
            "initialized": self._initialized,
            "documents": (
                self._vector_store.count()
                if self._vector_store and self._initialized else 0
            ),
        }

    def checkpoint(self) -> None:
        raise NotImplementedError("Checkpointing coming soon")

    def __repr__(self):
        return (
            f"Docqwise(strategy='{self._config.strategy.mode}', "
            f"store='{self._config.storage.vector_store}', "
            f"workers={self._config.speed.workers})"
        )
