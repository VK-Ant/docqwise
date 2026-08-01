"""Base reader abstract class."""

from abc import ABC, abstractmethod

from docqwise.core.document import DocqwiseDocument


class BaseReader(ABC):
    """Abstract base class for document readers. Implement to add new format support."""

    @abstractmethod
    def can_read(self, path: str) -> bool:
        """Return True if this reader can handle the given path or format."""

    @abstractmethod
    def read(self, path: str, **kwargs) -> DocqwiseDocument:
        """Read a document and return a DocqwiseDocument."""

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions (e.g. ['.pdf', '.PDF'])."""
