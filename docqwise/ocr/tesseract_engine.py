"""Tesseract OCR engine."""
from __future__ import annotations
import numpy as np
from docqwise.ocr.base import BaseOCREngine, OCRResult, OCRWord

class TesseractEngine(BaseOCREngine):
    def __init__(self, lang: str = "eng"):
        self._lang = lang

    def ocr_page(self, image: np.ndarray, language: str = "en") -> OCRResult:
        import pytesseract
        text = pytesseract.image_to_string(image, lang=self._lang)
        data = pytesseract.image_to_data(image, lang=self._lang, output_type=pytesseract.Output.DICT)
        words = []
        for i, word in enumerate(data["text"]):
            if word.strip():
                from docqwise.core.chunk import BoundingBox
                bbox = BoundingBox(data["left"][i], data["top"][i],
                                   data["left"][i] + data["width"][i],
                                   data["top"][i] + data["height"][i])
                conf = float(data["conf"][i]) / 100.0 if data["conf"][i] != -1 else 0.5
                words.append(OCRWord(text=word, bbox=bbox, confidence=conf))
        avg_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
        return OCRResult(text=text.strip(), words=words, confidence=avg_conf, language=language)

    def ocr_batch(self, images: list[np.ndarray], **kwargs) -> list[OCRResult]:
        return [self.ocr_page(img, **kwargs) for img in images]

    def supported_languages(self) -> list[str]:
        return ["eng", "fra", "deu", "spa", "ita", "por", "hin", "tam"]
