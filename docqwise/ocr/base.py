"""Base OCR engine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from docqwise.core.chunk import BoundingBox

@dataclass
class OCRWord:
    text: str
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0

@dataclass
class OCRResult:
    text: str
    words: list[OCRWord] = field(default_factory=list)
    confidence: float = 0.0
    language: str = ""

class BaseOCREngine(ABC):
    @abstractmethod
    def ocr_page(self, image: np.ndarray, language: str = "en") -> OCRResult: ...
    @abstractmethod
    def ocr_batch(self, images: list[np.ndarray], **kwargs) -> list[OCRResult]: ...
    @abstractmethod
    def supported_languages(self) -> list[str]: ...
