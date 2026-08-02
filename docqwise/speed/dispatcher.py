"""Batch dispatcher with multiprocessing support."""
from __future__ import annotations
import os, time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

class BatchDispatcher:
    def __init__(self, workers: int = 1, threads: int = 4):
        self.workers = workers
        self.threads = threads

    def dispatch(self, items: list, process_fn, show_progress: bool = True,
                 batch_size: int = 1, max_retries: int = 1) -> list:
        results = []
        failed = []
        executor_cls = ProcessPoolExecutor if self.workers > 1 else ThreadPoolExecutor
        max_w = self.workers if self.workers > 1 else self.threads
        iterator = tqdm(total=len(items), desc="Processing", disable=not show_progress)
        with executor_cls(max_workers=max_w) as executor:
            futures = {executor.submit(process_fn, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    failed.append({"item": str(item), "error": str(e)})
                iterator.update(1)
        iterator.close()
        return {"results": results, "failed": failed, "total": len(items),
                "success": len(results), "errors": len(failed)}

    def dispatch_async(self, items, process_fn, **kwargs):
        import asyncio
        async def _run():
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, process_fn, item) for item in items]
            return await asyncio.gather(*tasks, return_exceptions=True)
        return asyncio.run(_run())
