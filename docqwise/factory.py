"""Factory system for docqwise.

All component selection logic lives here. Engine.py calls factories.
Adding a new extractor, LLM, store, or embedder = register it here.
Engine.py NEVER changes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("docqwise")


# ══════════════════════════════════════════
# EXTRACTOR FACTORY
# ══════════════════════════════════════════

class ExtractorFactory:
    """Creates the right extractor based on method.
    
    Adding a new method:
        1. Create docqwise/extractors/my_extractor.py
        2. Register it in EXTRACTORS dict below
        3. Done. Engine.py doesn't change.
    """

    EXTRACTORS = {
        "rag": "docqwise.extractors.rag_extractor.RAGExtractor",
        "llm": "docqwise.extractors.llm_extractor.LLMFieldExtractor",
        "regex": "docqwise.extractors.field_extractor.RegexFieldExtractor",
        "vision": "docqwise.extractors.multimodal_extractor.MultimodalExtractor",
    }

    @classmethod
    def get(cls, method: str = "auto", llm=None, embedder=None,
            model: str = None, **kwargs) -> Any:
        """Get extractor instance for the given method."""
        if method == "auto":
            method = "rag"  # RAG is the default

        path = cls.EXTRACTORS.get(method)
        if not path:
            raise ValueError(f"Unknown extraction method: {method}. "
                             f"Available: {list(cls.EXTRACTORS.keys())}")

        module_path, class_name = path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        extractor_class = getattr(module, class_name)

        # Pass relevant kwargs based on extractor type
        if method == "rag":
            return extractor_class(llm=llm, embedder=embedder, model=model,
                                   chunk_top_k=kwargs.get("top_k", 5))
        elif method == "llm":
            return extractor_class(llm=llm, model=model)
        elif method == "vision":
            return extractor_class(model=model or "gpt-4o-mini",
                                   api_key=kwargs.get("api_key"))
        elif method == "regex":
            return extractor_class()
        else:
            return extractor_class()

    @classmethod
    def register(cls, method: str, class_path: str):
        """Register a new extraction method."""
        cls.EXTRACTORS[method] = class_path

    @classmethod
    def available_methods(cls) -> list[str]:
        return list(cls.EXTRACTORS.keys())


# ══════════════════════════════════════════
# LLM FACTORY
# ══════════════════════════════════════════

class LLMFactory:
    """Creates LLM instance. Auto-detects what's available.
    
    Priority: user-provided → Ollama → HuggingFace → OpenAI
    """

    @classmethod
    def get(cls, llm=None, model: str = None) -> Optional[Any]:
        """Get the best available LLM."""
        if llm is not None:
            return llm

        # Ollama (local, free, fast startup)
        try:
            import requests
            requests.get("http://localhost:11434/api/tags", timeout=2)
            from docqwise.llm.ollama import OllamaLLM
            return OllamaLLM(model=model or "nemotron-mini")
        except Exception:
            pass

        # HuggingFace (local, free, needs GPU)
        try:
            import torch
            if torch.cuda.is_available():
                from docqwise.llm.hf_llm import HuggingFaceLLM
                return HuggingFaceLLM(
                    model_name=model or "Qwen/Qwen2.5-3B-Instruct",
                    quantize="4bit",
                )
        except ImportError:
            pass

        # OpenAI (cloud, paid)
        try:
            import os
            if os.environ.get("OPENAI_API_KEY"):
                from docqwise.llm.openai_llm import OpenAILLM
                return OpenAILLM(model=model or "gpt-4o-mini")
        except Exception:
            pass

        return None

    @classmethod
    def available(cls) -> list[str]:
        """List available LLM backends."""
        backends = []
        try:
            import requests
            requests.get("http://localhost:11434/api/tags", timeout=2)
            backends.append("ollama")
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                backends.append("huggingface")
        except ImportError:
            pass
        try:
            import os
            if os.environ.get("OPENAI_API_KEY"):
                backends.append("openai")
        except Exception:
            pass
        return backends


# ══════════════════════════════════════════
# EMBEDDER FACTORY
# ══════════════════════════════════════════

class EmbedderFactory:
    """Creates embedder instance."""

    @classmethod
    def get(cls, embedder=None, model: str = None) -> Optional[Any]:
        if embedder is not None:
            return embedder
        try:
            from docqwise.embedders.sentence_transformer import SentenceTransformerEmbedder
            return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
        except ImportError:
            return None


# ══════════════════════════════════════════
# STORE FACTORY
# ══════════════════════════════════════════

class StoreFactory:
    """Creates vector store instance."""

    STORES = {
        "sqlite": "docqwise.stores.sqlite_store.SQLiteVectorStore",
        "faiss": "docqwise.stores.faiss_store.FAISSVectorStore",
    }

    @classmethod
    def get(cls, store=None, store_type: str = "sqlite",
            store_path: str = "./docqwise_db") -> Any:
        if store is not None and hasattr(store, "insert"):
            return store

        path = cls.STORES.get(store_type, cls.STORES["sqlite"])
        module_path, class_name = path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        store_class = getattr(module, class_name)
        return store_class(store_path)

    @classmethod
    def register(cls, name: str, class_path: str):
        cls.STORES[name] = class_path


# ══════════════════════════════════════════
# CHUNKER FACTORY
# ══════════════════════════════════════════

class ChunkerFactory:
    """Creates chunker instance."""

    CHUNKERS = {
        "structure": "docqwise.chunkers.structure_chunker.StructureChunker",
        "fixed": "docqwise.chunkers.fixed_chunker.FixedChunker",
        "sentence": "docqwise.chunkers.sentence_chunker.SentenceChunker",
    }

    @classmethod
    def get(cls, chunker=None, strategy: str = "structure", **kwargs) -> Any:
        if chunker is not None:
            return chunker
        path = cls.CHUNKERS.get(strategy, cls.CHUNKERS["structure"])
        module_path, class_name = path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**kwargs)

    @classmethod
    def register(cls, name: str, class_path: str):
        cls.CHUNKERS[name] = class_path


# ══════════════════════════════════════════
# TEMPLATE FACTORY
# ══════════════════════════════════════════

class TemplateFactory:
    """Creates template instance."""

    TEMPLATES = {
        "invoice": "docqwise.templates.invoice.InvoiceTemplate",
        "contract": "docqwise.templates.contract.ContractTemplate",
        "resume": "docqwise.templates.resume.ResumeTemplate",
        "receipt": "docqwise.templates.receipt.ReceiptTemplate",
    }

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        path = cls.TEMPLATES.get(name)
        if not path:
            return None
        module_path, class_name = path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()

    @classmethod
    def register(cls, name: str, class_path: str):
        cls.TEMPLATES[name] = class_path

    @classmethod
    def available(cls) -> list[str]:
        return list(cls.TEMPLATES.keys())
