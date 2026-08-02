"""Hash-based change detection for incremental processing."""
from __future__ import annotations
import hashlib, os, sqlite3

class ChangeDetector:
    def __init__(self, store_path: str = "./docqwise_db"):
        os.makedirs(store_path, exist_ok=True)
        self._db_path = os.path.join(store_path, "file_hashes.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS file_hashes (
                path TEXT PRIMARY KEY, hash TEXT, size INTEGER, mtime REAL, processed_at TEXT
            )""")
            conn.commit()
        finally:
            conn.close()

    def compute_hash(self, path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def has_changed(self, path: str) -> bool:
        path = os.path.abspath(path)
        stat = os.stat(path)
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT hash, size, mtime FROM file_hashes WHERE path = ?", (path,)).fetchone()
        finally:
            conn.close()
        if not row:
            return True
        stored_hash, stored_size, stored_mtime = row
        if stat.st_size != stored_size or stat.st_mtime != stored_mtime:
            return True
        return self.compute_hash(path) != stored_hash

    def mark_processed(self, path: str):
        path = os.path.abspath(path)
        stat = os.stat(path)
        file_hash = self.compute_hash(path)
        from datetime import datetime
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes (path, hash, size, mtime, processed_at) VALUES (?,?,?,?,?)",
                (path, file_hash, stat.st_size, stat.st_mtime, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_changed_files(self, directory: str, extensions: list[str] = None) -> list[str]:
        changed = []
        for root, dirs, files in os.walk(directory):
            for fname in files:
                if extensions and not any(fname.lower().endswith(ext) for ext in extensions):
                    continue
                fpath = os.path.join(root, fname)
                if self.has_changed(fpath):
                    changed.append(fpath)
        return changed
