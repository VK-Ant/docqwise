"""Progress tracking."""
from __future__ import annotations
import time

class ProgressTracker:
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.done = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.done += n

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        return self.done / max(self.elapsed, 0.001)

    @property
    def eta(self) -> float:
        remaining = self.total - self.done
        return remaining / max(self.rate, 0.001)

    def __str__(self):
        pct = self.done / max(self.total, 1) * 100
        return f"{self.desc}: {self.done}/{self.total} ({pct:.0f}%) | {self.rate:.1f}/sec | ETA: {self.eta:.0f}s"
