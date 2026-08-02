"""Base vector store."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from docqwise.core.chunk import DocqwiseChunk

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
    @abstractmethod
    def get_all_doc_ids(self) -> list[str]: ...
