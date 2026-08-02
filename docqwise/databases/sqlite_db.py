"""SQLite database connector."""
from __future__ import annotations
import sqlite3
from docqwise.databases.base import BaseDatabaseStore

class SQLiteDB(BaseDatabaseStore):
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row

    def connect(self, connection_string: str, **kwargs):
        self._conn = sqlite3.connect(connection_string.replace("sqlite:///", ""))
        self._conn.row_factory = sqlite3.Row

    def insert(self, data: dict, table: str):
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        self._conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", list(data.values()))
        self._conn.commit()

    def query(self, query: str, params=None) -> list[dict]:
        cursor = self._conn.execute(query, params or [])
        return [dict(row) for row in cursor.fetchall()]

    def schema(self, table=None) -> dict:
        if table:
            rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            return {"table": table, "columns": [{"name": r["name"], "type": r["type"]} for r in rows]}
        tables = self.tables()
        return {t: self.schema(t) for t in tables}

    def tables(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        return [r["name"] for r in rows]
