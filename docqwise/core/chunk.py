"""Chunk representation for retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADING = "heading"
    LIST = "list"
    CAPTION = "caption"
    FORM_FIELD = "form_field"
    CODE = "code"
    EQUATION = "equation"


@dataclass
class BoundingBox:
    """Bounding box coordinates on a page."""

    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class ChunkMetadata:
    """Metadata attached to a chunk."""

    doc_id: str = ""
    source_path: str = ""
    page_num: int = 0
    chunk_index: int = 0
    element_type: ElementType = ElementType.TEXT
    heading_context: str = ""
    section_path: list[str] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocqwiseChunk:
    """A retrieval-ready piece of a document."""

    chunk_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # For table chunks
    table_data: Optional[dict] = None

    # For image chunks
    image_path: Optional[str] = None
    image_description: str = ""
