"""Base embedder."""
from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...
    @abstractmethod
    def dimension(self) -> int: ...
