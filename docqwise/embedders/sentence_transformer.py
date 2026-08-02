"""SentenceTransformer embedder (default)."""
from __future__ import annotations
import numpy as np
from docqwise.embedders.base import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dim = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        return self._model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    def dimension(self) -> int:
        self._ensure_model()
        return self._dim
