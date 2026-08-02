"""Base LLM interface."""
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    def generate_structured(self, prompt: str, schema: dict) -> dict: ...
