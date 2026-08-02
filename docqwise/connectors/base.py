"""Base connector."""
from abc import ABC, abstractmethod
from typing import Any

class BaseConnector(ABC):
    @abstractmethod
    def list_files(self, path: str, **kwargs) -> list[str]: ...
    @abstractmethod
    def read_file(self, path: str) -> bytes: ...
