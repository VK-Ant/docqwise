"""Field extraction result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from docqwise.core.chunk import BoundingBox


@dataclass
class FieldResult:
    """Result of a single field extraction."""

    value: Any
    confidence: float
    raw_text: str = ""
    bbox: Optional[BoundingBox] = None
    page: int = 0
    extraction_method: str = ""
    is_corrected: bool = False


@dataclass
class ExtractionResult:
    """Result of extracting fields from a document."""

    doc_id: str
    source_path: str
    fields: dict[str, FieldResult]
    confidence: float
    strategy_used: str = ""
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    _engine: Any = field(default=None, repr=False)

    def correct(self, corrections: dict[str, Any], scope: str = "global") -> None:
        """Apply user corrections and store for future use."""
        for field_name, correct_value in corrections.items():
            if field_name in self.fields:
                self.fields[field_name] = FieldResult(
                    value=correct_value,
                    confidence=1.0,
                    raw_text=self.fields[field_name].raw_text,
                    bbox=self.fields[field_name].bbox,
                    page=self.fields[field_name].page,
                    extraction_method="user_correction",
                    is_corrected=True,
                )
            else:
                self.fields[field_name] = FieldResult(
                    value=correct_value,
                    confidence=1.0,
                    extraction_method="user_correction",
                    is_corrected=True,
                )

        if self._engine is not None:
            self._engine._correction_store.save(
                doc_id=self.doc_id,
                source_path=self.source_path,
                corrections=corrections,
                scope=scope,
            )

    def to_dict(self) -> dict[str, Any]:
        return {k: v.value for k, v in self.fields.items()}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), default=str, indent=2)
