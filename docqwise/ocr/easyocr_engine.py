"""EasyOCR engine (default)."""
from __future__ import annotations
import numpy as np
from docqwise.ocr.base import BaseOCREngine, OCRResult, OCRWord
from docqwise.core.chunk import BoundingBox

class EasyOCREngine(BaseOCREngine):
    def __init__(self, languages: list[str] = None, gpu: bool = True):
        self._languages = languages or ["en"]
        self._gpu = gpu
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages, gpu=self._gpu)

    def ocr_page(self, image: np.ndarray, language: str = "en") -> OCRResult:
        self._ensure_reader()
        results = self._reader.readtext(image)
        words = []
        all_text = []
        total_conf = 0.0
        for bbox_pts, text, conf in results:
            x_coords = [p[0] for p in bbox_pts]
            y_coords = [p[1] for p in bbox_pts]
            bbox = BoundingBox(min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            words.append(OCRWord(text=text, bbox=bbox, confidence=conf))
            all_text.append(text)
            total_conf += conf
        avg_conf = total_conf / len(results) if results else 0.0
        return OCRResult(text=" ".join(all_text), words=words, confidence=avg_conf, language=language)

    def ocr_batch(self, images: list[np.ndarray], **kwargs) -> list[OCRResult]:
        return [self.ocr_page(img, **kwargs) for img in images]

    def supported_languages(self) -> list[str]:
        return self._languages
