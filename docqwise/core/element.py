"""Document element types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from docqwise.core.chunk import BoundingBox, DocqwiseChunk, ChunkMetadata, ElementType


@dataclass
class TableCell:
    """A single cell in a table."""

    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0


@dataclass
class Table:
    """Extracted table from a document."""

    table_id: str
    cells: list[TableCell]
    rows: int
    cols: int
    page: int = 0
    bbox: Optional[BoundingBox] = None
    caption: str = ""
    confidence: float = 1.0
    context: str = ""
    has_merged_cells: bool = False
    is_cross_page: bool = False

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd

        grid = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        headers = []

        for cell in self.cells:
            if cell.row < self.rows and cell.col < self.cols:
                grid[cell.row][cell.col] = cell.text
                if cell.is_header:
                    headers.append(cell.text)

        if headers:
            return pd.DataFrame(grid[1:], columns=grid[0])
        return pd.DataFrame(grid)

    def to_csv(self, path: str = "") -> str:
        """Export as CSV."""
        df = self.to_dataframe()
        if path:
            df.to_csv(path, index=False)
            return path
        return df.to_csv(index=False)

    def to_json(self) -> list[dict]:
        """Export as list of row dicts."""
        df = self.to_dataframe()
        return df.to_dict(orient="records")

    def to_excel(self, path: str) -> None:
        """Export as Excel file."""
        df = self.to_dataframe()
        df.to_excel(path, index=False)

    def to_markdown(self) -> str:
        """Export as markdown table."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def to_chunk(self) -> DocqwiseChunk:
        """Convert to retrieval chunk with table context."""
        text = f"Table: {self.caption}\n{self.to_markdown()}" if self.caption else self.to_markdown()
        return DocqwiseChunk(
            chunk_id=f"table_{self.table_id}",
            text=text,
            metadata=ChunkMetadata(
                page_num=self.page,
                element_type=ElementType.TABLE,
                bbox=self.bbox,
                confidence=self.confidence,
            ),
            table_data=self.to_json(),
        )


@dataclass
class ExtractedImage:
    """Image or figure extracted from a document."""

    image_id: str
    page: int = 0
    bbox: Optional[BoundingBox] = None
    caption: str = ""
    alt_text: str = ""
    image_type: str = ""  # photo | chart | diagram | logo | signature
    chart_type: str = ""  # bar | line | pie | scatter
    chart_data: Any = None
    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    resolution: tuple[int, int] = (0, 0)
    confidence: float = 1.0

    def save(self, directory: str, filename: str = "") -> str:
        """Save extracted image to file."""
        import os

        if not filename:
            filename = f"image_{self.image_id}.png"
        path = os.path.join(directory, filename)
        if self.image_bytes:
            os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as f:
                f.write(self.image_bytes)
        return path


@dataclass
class Entity:
    """Named entity extracted from a document."""

    text: str
    entity_type: str  # PERSON, ORG, MONEY, DATE, LOCATION, custom
    page: int = 0
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    normalized_value: Any = None


@dataclass
class Relation:
    """Relation between two entities."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    page: int = 0
    source_entities: list[Entity] = field(default_factory=list)


@dataclass
class FormField:
    """Form field extracted from a document."""

    field_name: str
    field_type: str  # text | checkbox | radio | dropdown | signature
    value: Any = None
    is_checked: Optional[bool] = None
    options: list[str] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    page: int = 0
    confidence: float = 1.0
