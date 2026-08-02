"""File watcher for continuous ingestion."""
from __future__ import annotations
import os, time, threading

class FileWatcher:
    def __init__(self, path: str, callback, interval: int = 60, extensions: list[str] = None):
        self.path = path
        self.callback = callback
        self.interval = interval
        self.extensions = extensions
        self._running = False
        self._thread = None
        self._known_files = set()

    def start(self):
        self._running = True
        self._known_files = self._scan()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        while self._running:
            time.sleep(self.interval)
            current = self._scan()
            new_files = current - self._known_files
            if new_files:
                for f in new_files:
                    self.callback(f)
                self._known_files = current

    def _scan(self) -> set:
        files = set()
        if os.path.isdir(self.path):
            for root, _, fnames in os.walk(self.path):
                for fname in fnames:
                    if self.extensions and not any(fname.endswith(e) for e in self.extensions):
                        continue
                    files.add(os.path.join(root, fname))
        return files
