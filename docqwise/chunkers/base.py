"""Base chunker."""
from abc import ABC, abstractmethod
from docqwise.core.document import DocqwiseDocument
from docqwise.core.chunk import DocqwiseChunk

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: DocqwiseDocument, **kwargs) -> list[DocqwiseChunk]: ...
