"""Prefetch sampled batches in worker processes for training loops.

This helper hides multiprocessing queue management behind a small, script-friendly interface.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable, Iterable, Sequence, Any


class Prefetcher:
    """Simple multiprocessing prefetcher for numpy-based batch sampling."""

    def __init__(
        self,
        worker_fn: Callable[..., None],
        worker_args: Sequence[Any],
        num_workers: int = 2,
        maxsize: int = 4,
        seed: int = 0,
    ) -> None:
        self.ctx = mp.get_context("spawn")
        self.queue = self.ctx.Queue(maxsize=maxsize)
        self.stop_event = self.ctx.Event()
        self.workers = []
        self._seed = int(seed)

        for i in range(num_workers):
            args = (self.queue, self.stop_event, self._seed + i * 9973, *worker_args)
            p = self.ctx.Process(target=worker_fn, args=args)
            p.daemon = True
            p.start()
            self.workers.append(p)

    def get(self):
        return self.queue.get()

    def close(self) -> None:
        if self.stop_event is None:
            return
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=1.0)
        try:
            self.queue.close()
        except Exception:
            pass
