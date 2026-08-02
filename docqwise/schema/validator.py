"""Schema validation."""
from __future__ import annotations
import re

class SchemaValidator:
    def validate(self, rows: list[dict], schema: dict) -> dict:
        errors = []
        warnings = []
        for i, row in enumerate(rows):
            for field_spec in schema.get("fields", []):
                name = field_spec["name"]
                val = row.get(name, "")
                if not val and not field_spec.get("nullable", True):
                    errors.append(f"Row {i}: '{name}' is required but empty")
                if val and field_spec["type"] == "number":
                    try:
                        float(str(val).replace(",", ""))
                    except ValueError:
                        errors.append(f"Row {i}: '{name}' expected number, got '{val}'")
                if val and field_spec["type"] == "date":
                    if not re.match(r"\d{4}-\d{2}-\d{2}", str(val)):
                        warnings.append(f"Row {i}: '{name}' may not be a valid date: '{val}'")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings,
                "rows_checked": len(rows), "fields_checked": len(schema.get("fields", []))}
