"""All pluggable base classes for docqwise.

Every component is a Base* class. Swap anything. Test anything. Deploy anywhere.
Import from the specific module (e.g. docqwise.ocr.base) for implementation.
This file serves as a single reference for all interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk, BoundingBox
from docqwise.core.field import ExtractionResult
from docqwise.core.element import Table, Entity, Relation, FormField


# ── PROCESSING LAYER ──


class BaseReader(ABC):
    @abstractmethod
    def can_read(self, path: str) -> bool: ...
    @abstractmethod
    def read(self, path: str, **kwargs) -> DocqwiseDocument: ...
    @abstractmethod
    def supported_formats(self) -> list[str]: ...


class BaseOCREngine(ABC):
    @abstractmethod
    def ocr_page(self, image: np.ndarray, language: str = "en") -> dict: ...
    @abstractmethod
    def ocr_batch(self, images: list[np.ndarray], **kwargs) -> list[dict]: ...
    @abstractmethod
    def supported_languages(self) -> list[str]: ...


class BaseLayoutModel(ABC):
    @abstractmethod
    def detect_layout(self, image: np.ndarray) -> list[dict]: ...
    @abstractmethod
    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]: ...


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, document: DocqwiseDocument, **kwargs) -> DocqwiseDocument: ...


class BaseTableExtractor(ABC):
    @abstractmethod
    def extract_tables(self, document: DocqwiseDocument, **kwargs) -> list[Table]: ...


class BaseFormExtractor(ABC):
    @abstractmethod
    def extract_form(self, document: DocqwiseDocument) -> dict[str, FormField]: ...


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: DocqwiseDocument, **kwargs) -> list[DocqwiseChunk]: ...


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...
    @abstractmethod
    def dimension(self) -> int: ...


# ── STORAGE LAYER ──


class BaseVectorStore(ABC):
    @abstractmethod
    def insert(self, chunks: list[DocqwiseChunk]) -> None: ...
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[dict] = None) -> list[dict]: ...
    @abstractmethod
    def delete(self, doc_id: str) -> None: ...
    @abstractmethod
    def update(self, doc_id: str, chunks: list[DocqwiseChunk]) -> None: ...
    @abstractmethod
    def count(self) -> int: ...


class BaseDatabaseStore(ABC):
    @abstractmethod
    def connect(self, connection_string: str, **kwargs) -> None: ...
    @abstractmethod
    def insert(self, data: dict, table: str) -> None: ...
    @abstractmethod
    def query(self, query: str, params: Optional[dict] = None) -> list[dict]: ...
    @abstractmethod
    def schema(self, table: Optional[str] = None) -> dict: ...
    @abstractmethod
    def tables(self) -> list[str]: ...


class BaseGraphStore(ABC):
    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None: ...
    @abstractmethod
    def add_edge(self, from_id: str, to_id: str, relation: str,
                 properties: Optional[dict] = None) -> None: ...
    @abstractmethod
    def query(self, query: str) -> list[dict]: ...
    @abstractmethod
    def neighbors(self, node_id: str, hops: int = 1) -> list[dict]: ...
    @abstractmethod
    def communities(self) -> list[list[str]]: ...


# ── INTELLIGENCE LAYER ──


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    def generate_structured(self, prompt: str, schema: dict) -> dict: ...


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str],
               top_k: int = 5) -> list[dict]: ...


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, document: DocqwiseDocument,
                 labels: Optional[list[str]] = None) -> list[dict]: ...


class BaseComparator(ABC):
    @abstractmethod
    def compare(self, doc_a: DocqwiseDocument,
                doc_b: DocqwiseDocument) -> dict: ...


# ── INFRASTRUCTURE LAYER ──


class BaseConnector(ABC):
    @abstractmethod
    def list_files(self, path: str, **kwargs) -> list[str]: ...
    @abstractmethod
    def read_file(self, path: str) -> bytes: ...
    @abstractmethod
    def watch(self, path: str, callback: Any, interval: int = 60) -> None: ...


class BaseCacheLayer(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]: ...
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
    @abstractmethod
    def invalidate(self, key: str) -> None: ...


class BaseExporter(ABC):
    @abstractmethod
    def export(self, documents: list[DocqwiseDocument], path: str, **kwargs) -> None: ...


class BaseTemplate(ABC):
    @abstractmethod
    def schema(self) -> dict: ...
    @abstractmethod
    def extract(self, document: DocqwiseDocument) -> ExtractionResult: ...
    @abstractmethod
    def validate(self, result: ExtractionResult) -> dict: ...
