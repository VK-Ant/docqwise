"""Auto schema detection."""
from __future__ import annotations
import re

class SchemaDetector:
    TYPE_MAP = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}

    def detect(self, headers: list[str], rows: list[dict]) -> dict:
        schema = {"fields": []}
        for header in headers:
            values = [r.get(header, "") for r in rows[:20] if r.get(header)]
            field_type = self._infer_type(values)
            nullable = any(not r.get(header) for r in rows[:20])
            schema["fields"].append({"name": header, "type": field_type, "nullable": nullable})
        return schema

    def _infer_type(self, values: list) -> str:
        if not values:
            return "string"
        for val in values:
            try:
                int(str(val).replace(",", ""))
                continue
            except ValueError:
                break
        else:
            return "integer"
        for val in values:
            try:
                float(str(val).replace(",", ""))
                continue
            except ValueError:
                break
        else:
            return "number"
        date_pattern = r"\d{4}-\d{2}-\d{2}"
        if all(re.match(date_pattern, str(v)) for v in values):
            return "date"
        if all(str(v).lower() in ("true", "false", "yes", "no", "1", "0") for v in values):
            return "boolean"
        return "string"
