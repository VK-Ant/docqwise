"""Benchmark runner."""
from __future__ import annotations
import time

class BenchmarkRunner:
    def run(self, documents: list[str], extractor, **kwargs) -> dict:
        results = []
        total_time = 0
        for doc_path in documents:
            start = time.time()
            try:
                result = extractor(doc_path)
                elapsed = time.time() - start
                results.append({"path": doc_path, "status": "success", "time_ms": elapsed * 1000})
                total_time += elapsed
            except Exception as e:
                results.append({"path": doc_path, "status": "error", "error": str(e)})
        return {
            "total_documents": len(documents),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "total_time_s": round(total_time, 2),
            "avg_time_ms": round(total_time / max(len(documents), 1) * 1000, 1),
            "results": results,
        }
