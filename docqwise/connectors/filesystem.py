"""Local filesystem connector."""
from __future__ import annotations
import os

class FilesystemConnector:
    def list_files(self, path: str, extensions: list[str] = None, recursive: bool = True) -> list[str]:
        files = []
        if os.path.isfile(path):
            return [path]
        walker = os.walk(path) if recursive else [(path, [], os.listdir(path))]
        for root, dirs, fnames in walker:
            for fname in fnames:
                if extensions and not any(fname.lower().endswith(ext) for ext in extensions):
                    continue
                files.append(os.path.join(root, fname))
        return sorted(files)

    def read_file(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()
