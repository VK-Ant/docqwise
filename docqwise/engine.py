"""Main Docqwise engine class."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from docqwise.config import DocqwiseConfig
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk
from docqwise.core.field import ExtractionResult
from docqwise.core.element import Table, ExtractedImage, Entity, Relation

logger = logging.getLogger("docqwise")


class Docqwise:
    """
    DocQWise: Read. Extract. Retrieve.

    Document intelligence engine that reads any format, extracts exact
    structured data, and retrieves with semantic search.

    Usage:
        from docqwise import Docqwise

        dq = Docqwise()
        dq.ingest("documents/")
        results = dq.retrieve("payment terms", top_k=5)
        answer = dq.ask("What is the total amount?")
    """

    def __init__(
        self,
        # Strategy
        strategy: str = "auto",
        confidence_threshold: float = 0.85,
        max_api_cost_per_doc: float = 0.05,
        # Speed
        workers: Union[int, str] = 1,
        gpu_workers: int = 0,
        threads: int = 4,
        batch_size: int = 32,
        prefetch: bool = False,
        async_mode: bool = False,
        # Models
        ocr: Any = "easyocr",
        layout_model: Any = "yolov8",
        embedder: Any = "all-MiniLM-L6-v2",
        llm: Any = None,
        reranker: Any = None,
        # Storage
        vector_store: Any = "sqlite",
        database: Any = None,
        graph_store: Any = None,
        store_path: str = "./docqwise_db",
        # Search
        search_mode: str = "vector",
        # Incremental
        incremental: bool = True,
        cache: bool = True,
        cache_dir: str = ".docqwise_cache",
        # Learning
        learning_mode: str = "exact",
        # Config override
        config: Optional[Union[str, DocqwiseConfig]] = None,
    ):
        if isinstance(config, str):
            self._config = DocqwiseConfig.from_yaml(config)
        elif isinstance(config, DocqwiseConfig):
            self._config = config
        else:
            self._config = DocqwiseConfig()
            self._config.strategy.mode = strategy
            self._config.strategy.confidence_threshold = confidence_threshold
            self._config.speed.workers = workers
            self._config.speed.gpu_workers = gpu_workers
            self._config.speed.threads = threads
            self._config.speed.batch_size = batch_size
            self._config.speed.prefetch = prefetch
            self._config.speed.async_mode = async_mode
            self._config.storage.vector_store = vector_store if isinstance(vector_store, str) else "custom"
            self._config.storage.store_path = store_path
            self._config.storage.cache = cache
            self._config.storage.cache_dir = cache_dir
            self._config.search.mode = search_mode
            self._config.incremental.enabled = incremental

        # Store raw component references for lazy init
        self._ocr_ref = ocr
        self._layout_ref = layout_model
        self._embedder_ref = embedder
        self._llm_ref = llm
        self._reranker_ref = reranker
        self._vector_store_ref = vector_store
        self._database_ref = database
        self._graph_store_ref = graph_store
        self._learning_mode = learning_mode

        # Lazy-initialized components
        self._reader = None
        self._ocr_engine = None
        self._layout_model = None
        self._embedder = None
        self._llm = None
        self._reranker = None
        self._vector_store = None
        self._database = None
        self._graph_store = None
        self._correction_store = None
        self._strategy_router = None
        self._change_detector = None

        self._initialized = False
        logger.info("DocQWise engine created (lazy init)")

    def _ensure_initialized(self) -> None:
        """Lazy initialization of components on first use."""
        if self._initialized:
            return
        self._init_reader()
        self._init_store()
        self._init_correction_store()
        self._initialized = True
        logger.info("DocQWise engine initialized")

    def _init_reader(self) -> None:
        from docqwise.readers.auto_reader import AutoReader
        self._reader = AutoReader()

    def _init_store(self) -> None:
        if isinstance(self._vector_store_ref, str) and self._vector_store_ref == "sqlite":
            from docqwise.stores.sqlite_store import SQLiteVectorStore
            self._vector_store = SQLiteVectorStore(self._config.storage.store_path)
        elif hasattr(self._vector_store_ref, "insert"):
            self._vector_store = self._vector_store_ref

    def _init_embedder(self) -> None:
        if self._embedder is not None:
            return
        if isinstance(self._embedder_ref, str):
            from docqwise.embedders.sentence_transformer import SentenceTransformerEmbedder
            self._embedder = SentenceTransformerEmbedder(self._embedder_ref)
        elif hasattr(self._embedder_ref, "embed"):
            self._embedder = self._embedder_ref

    def _init_correction_store(self) -> None:
        from docqwise.learning.correction import CorrectionStore
        self._correction_store = CorrectionStore(self._config.storage.store_path)

    # ═══════════════════════════════════════
    # INGESTION
    # ═══════════════════════════════════════

    def ingest(self, source: str, **kwargs) -> dict:
        """Ingest documents from file, folder, URL, or database connection string."""
        self._ensure_initialized()
        # TODO: implement full ingestion pipeline
        logger.info(f"Ingesting from: {source}")
        raise NotImplementedError("Ingestion pipeline coming in v0.1.0")

    async def ingest_async(self, source: str, **kwargs) -> dict:
        """Async ingestion with parallel processing."""
        self._ensure_initialized()
        raise NotImplementedError("Async ingestion coming in v0.1.0")

    def sync(self, source: str = None, mode: str = "incremental",
             interval: int = 300) -> None:
        """Sync with database — incremental by default."""
        raise NotImplementedError("Database sync coming in v0.1.0")

    def watch(self, path: str, poll_interval: int = 60,
              on_new: Any = None) -> None:
        """Watch folder for new documents."""
        raise NotImplementedError("File watching coming in v0.1.0")

    # ═══════════════════════════════════════
    # EXTRACTION
    # ═══════════════════════════════════════

    def extract_all(self, source: str, **kwargs) -> DocqwiseDocument:
        """Extract everything from a document."""
        self._ensure_initialized()
        raise NotImplementedError("Full extraction coming in v0.1.0")

    def extract_text(self, source: str, **kwargs) -> str:
        """Extract text from a document."""
        self._ensure_initialized()
        raise NotImplementedError("Text extraction coming in v0.1.0")

    def extract_fields(self, source: str, schema: dict = None,
                       template: str = None, **kwargs) -> ExtractionResult:
        """Extract fields — schema-driven or auto-detect."""
        self._ensure_initialized()
        raise NotImplementedError("Field extraction coming in v0.1.0")

    def extract_fields_batch(self, source: str, **kwargs) -> list[ExtractionResult]:
        """Batch field extraction across multiple documents."""
        self._ensure_initialized()
        raise NotImplementedError("Batch extraction coming in v0.1.0")

    def extract_tables(self, source: str, **kwargs) -> list[Table]:
        """Extract tables from a document."""
        self._ensure_initialized()
        raise NotImplementedError("Table extraction coming in v0.1.0")

    def extract_images(self, source: str, **kwargs) -> list[ExtractedImage]:
        """Extract images and figures from a document."""
        self._ensure_initialized()
        raise NotImplementedError("Image extraction coming in v0.1.0")

    def extract_form(self, source: str) -> dict:
        """Extract form fields."""
        self._ensure_initialized()
        raise NotImplementedError("Form extraction coming in v0.1.0")

    def extract_entities(self, source: str, types: list[str] = None,
                         custom_types: dict = None) -> list[Entity]:
        """Extract named entities."""
        self._ensure_initialized()
        raise NotImplementedError("Entity extraction coming in v0.1.0")

    def extract_relations(self, source: str) -> list[Relation]:
        """Extract entity relations."""
        self._ensure_initialized()
        raise NotImplementedError("Relation extraction coming in v0.1.0")

    def extract_layout(self, source: str) -> dict:
        """Extract full layout map with bounding boxes."""
        self._ensure_initialized()
        raise NotImplementedError("Layout extraction coming in v0.1.0")

    def auto_extract(self, source: str) -> ExtractionResult:
        """Auto-detect document type and extract all relevant fields."""
        self._ensure_initialized()
        raise NotImplementedError("Auto extraction coming in v0.1.0")

    # ═══════════════════════════════════════
    # RETRIEVAL + Q&A
    # ═══════════════════════════════════════

    def retrieve(self, query: str, top_k: int = 5,
                 mode: str = None, **kwargs) -> list[dict]:
        """Semantic search across ingested corpus."""
        self._ensure_initialized()
        raise NotImplementedError("Retrieval coming in v0.1.0")

    def ask(self, question: str, source: str = None, **kwargs) -> str:
        """Natural language Q&A."""
        self._ensure_initialized()
        raise NotImplementedError("Q&A coming in v0.1.0")

    def ask_batch(self, question: str = None, questions: list = None,
                  documents: str = None, **kwargs) -> list:
        """Batch Q&A across documents."""
        raise NotImplementedError("Batch Q&A coming in v0.1.0")

    def sql(self, query: str) -> list[dict]:
        """Run SQL directly against ingested structured data."""
        raise NotImplementedError("SQL queries coming in v0.1.0")

    # ═══════════════════════════════════════
    # CLASSIFICATION + COMPARISON
    # ═══════════════════════════════════════

    def classify(self, source: str, **kwargs) -> list[dict]:
        """Classify document type."""
        raise NotImplementedError("Classification coming in v0.1.0")

    def classify_batch(self, source: str, **kwargs) -> list:
        """Batch classification."""
        raise NotImplementedError("Batch classification coming in v0.1.0")

    def set_taxonomy(self, labels: list[str]) -> None:
        """Set custom classification taxonomy."""
        raise NotImplementedError("Taxonomy coming in v0.1.0")

    def train_classifier(self, examples: dict) -> None:
        """Few-shot classifier training."""
        raise NotImplementedError("Classifier training coming in v0.1.0")

    def compare(self, source_a: str, source_b: str) -> dict:
        """Compare two documents."""
        raise NotImplementedError("Comparison coming in v0.1.0")

    def compare_batch(self, pairs: list, **kwargs) -> list:
        """Batch comparison."""
        raise NotImplementedError("Batch comparison coming in v0.1.0")

    def find_duplicates(self, source: str, threshold: float = 0.85) -> list:
        """Find near-duplicate documents."""
        raise NotImplementedError("Deduplication coming in v0.1.0")

    # ═══════════════════════════════════════
    # SCHEMA
    # ═══════════════════════════════════════

    def detect_schema(self, source: str) -> dict:
        """Auto-detect schema from structured data."""
        raise NotImplementedError("Schema detection coming in v0.1.0")

    def validate(self, source: str, schema: dict) -> dict:
        """Validate data against schema."""
        raise NotImplementedError("Validation coming in v0.1.0")

    def discover_schema(self, connection_string: str) -> dict:
        """Discover database schema."""
        raise NotImplementedError("Schema discovery coming in v0.1.0")

    # ═══════════════════════════════════════
    # GRAPH
    # ═══════════════════════════════════════

    def build_graph(self, source: str = None) -> Any:
        """Build document graph from ingested corpus."""
        raise NotImplementedError("GraphRAG coming in v0.1.0")

    # ═══════════════════════════════════════
    # LEARNING
    # ═══════════════════════════════════════

    def learn_from_labels(self, documents: str, labels: str) -> None:
        """Bulk learning from labeled dataset."""
        raise NotImplementedError("Bulk learning coming in v0.1.0")

    def learning_report(self) -> dict:
        """Report on corrections and accuracy improvements."""
        raise NotImplementedError("Learning report coming in v0.1.0")

    def export_learnings(self, path: str) -> None:
        """Export learned corrections."""
        raise NotImplementedError("Export learnings coming in v0.1.0")

    def import_learnings(self, path: str) -> None:
        """Import corrections from another instance."""
        raise NotImplementedError("Import learnings coming in v0.1.0")

    # ═══════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════

    def to_dataframe(self, source: str, **kwargs):
        """Convert structured source to DataFrame."""
        raise NotImplementedError("DataFrame conversion coming in v0.1.0")

    def detect_pii(self, source: str) -> list:
        """Detect PII in document."""
        raise NotImplementedError("PII detection coming in v0.1.0")

    def redact(self, source: str, output: str, redact_types: list = None) -> None:
        """Redact PII."""
        raise NotImplementedError("PII redaction coming in v0.1.0")

    def checkpoint(self) -> None:
        """Save current processing state."""
        raise NotImplementedError("Checkpointing coming in v0.1.0")

    def metrics(self) -> dict:
        """Pipeline health and performance metrics."""
        return {
            "version": "0.1.0",
            "store_path": self._config.storage.store_path,
            "strategy": self._config.strategy.mode,
            "workers": self._config.speed.workers,
            "initialized": self._initialized,
        }

    def export(self, path: str, format: str = "json", **kwargs) -> None:
        """Export processed data."""
        raise NotImplementedError("Export coming in v0.1.0")

    def __repr__(self) -> str:
        return (
            f"Docqwise(strategy='{self._config.strategy.mode}', "
            f"store='{self._config.storage.vector_store}', "
            f"workers={self._config.speed.workers})"
        )
