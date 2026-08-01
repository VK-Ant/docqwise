"""Configuration system for docqwise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class StrategyConfig:
    mode: str = "auto"
    confidence_threshold: float = 0.85
    max_api_cost_per_doc: float = 0.05


@dataclass
class SpeedConfig:
    workers: Union[int, str] = 1
    gpu_workers: int = 0
    threads: int = 4
    batch_size: int = 32
    prefetch: bool = False
    async_mode: bool = False
    max_memory_gb: float = 0


@dataclass
class StorageConfig:
    vector_store: str = "sqlite"
    database: Optional[str] = None
    graph_store: Optional[str] = None
    store_path: str = "./docqwise_db"
    cache: bool = True
    cache_dir: str = ".docqwise_cache"


@dataclass
class ModelConfig:
    ocr: str = "easyocr"
    layout_model: str = "yolov8"
    embedder: str = "all-MiniLM-L6-v2"
    llm: Optional[str] = None
    reranker: Optional[str] = None


@dataclass
class SearchConfig:
    mode: str = "vector"
    default_top_k: int = 5
    rerank: bool = False
    rerank_top_k: int = 5


@dataclass
class IncrementalConfig:
    enabled: bool = True
    change_detection: str = "hash"
    checkpoint_interval: int = 100


@dataclass
class ChunkerConfig:
    strategy: str = "structure"
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    preserve_tables: bool = True
    attach_headings: bool = True
    attach_page_context: bool = True


@dataclass
class DocqwiseConfig:
    """Master configuration."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    incremental: IncrementalConfig = field(default_factory=IncrementalConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)

    @classmethod
    def from_yaml(cls, path: str) -> DocqwiseConfig:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> DocqwiseConfig:
        import json

        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> DocqwiseConfig:
        config = cls()
        if "strategy" in data:
            config.strategy = StrategyConfig(**data["strategy"])
        if "speed" in data:
            config.speed = SpeedConfig(**data["speed"])
        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])
        if "models" in data:
            config.models = ModelConfig(**data["models"])
        if "search" in data:
            config.search = SearchConfig(**data["search"])
        if "incremental" in data:
            config.incremental = IncrementalConfig(**data["incremental"])
        if "chunker" in data:
            config.chunker = ChunkerConfig(**data["chunker"])
        return config

    def to_yaml(self, path: str) -> None:
        import yaml
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        import json
        from dataclasses import asdict

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
