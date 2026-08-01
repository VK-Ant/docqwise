"""Simple correction store — exact overrides, no ML."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Optional


class CorrectionStore:
    """Stores user corrections as exact overrides. No RL. No weights. Deterministic."""

    def __init__(self, store_path: str = "./docqwise_db"):
        os.makedirs(store_path, exist_ok=True)
        self._db_path = os.path.join(store_path, "corrections.db")
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    source_path TEXT,
                    field_name TEXT NOT NULL,
                    wrong_value TEXT,
                    correct_value TEXT NOT NULL,
                    scope TEXT DEFAULT 'global',
                    vendor TEXT,
                    doc_type TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrections_field
                ON corrections(field_name, scope, vendor)
            """)

    def save(self, doc_id: str, source_path: str,
             corrections: dict[str, Any], scope: str = "global") -> None:
        """Save corrections as exact overrides."""
        with sqlite3.connect(self._db_path) as conn:
            for field_name, correct_value in corrections.items():
                conn.execute(
                    """INSERT INTO corrections
                    (doc_id, source_path, field_name, correct_value, scope, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (doc_id, source_path, field_name,
                     json.dumps(correct_value, default=str),
                     scope, datetime.now().isoformat()),
                )

    def apply(self, doc_type: str, fields: dict[str, Any],
              vendor: Optional[str] = None) -> dict[str, Any]:
        """Apply stored corrections as exact overrides. No scoring. No probability."""
        corrected = dict(fields)
        with sqlite3.connect(self._db_path) as conn:
            for field_name in fields:
                rows = conn.execute(
                    """SELECT correct_value FROM corrections
                    WHERE field_name = ? AND (scope = 'global' OR vendor = ?)
                    ORDER BY created_at DESC LIMIT 1""",
                    (field_name, vendor or ""),
                ).fetchall()
                if rows:
                    corrected[field_name] = json.loads(rows[0][0])
        return corrected

    def list_corrections(self, doc_type: Optional[str] = None) -> list[dict]:
        """List all stored corrections."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            if doc_type:
                rows = conn.execute(
                    "SELECT * FROM corrections WHERE doc_type = ? ORDER BY created_at DESC",
                    (doc_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM corrections ORDER BY created_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    def delete_correction(self, correction_id: int) -> None:
        """Remove a correction."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM corrections WHERE id = ?", (correction_id,))

    def count(self) -> int:
        """Count total corrections."""
        with sqlite3.connect(self._db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
