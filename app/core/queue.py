import asyncio
from typing import Any, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging import get_logger


@dataclass
class QueueItem:
    """Represents a queued task"""
    id: str
    coroutine_factory: Callable[[], Coroutine]
    future: asyncio.Future
    created_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0  # Lower = higher priority


class InferenceQueue:
    """
    Async queue for LLM inference requests.

    Provides backpressure control so the LLM isn't overwhelmed
    with concurrent requests. Processes requests sequentially
    with configurable concurrency.
    """

    def __init__(self, max_concurrent: int = 2, max_queue_size: int = 50):
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=max_queue_size)
        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._active_tasks = 0
        self._total_processed = 0
        self._total_errors = 0
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._logger = get_logger()

    async def start(self) -> None:
        """Start queue workers"""
        if self._running:
            return

        self._running = True
        for i in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)

        self._logger.info(
            f"Inference queue started: {self._max_concurrent} workers, "
            f"max queue size: {self._max_queue_size}"
        )

    async def stop(self) -> None:
        """Stop queue workers gracefully"""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._logger.info("Inference queue stopped")

    async def _worker(self, worker_name: str) -> None:
        """Worker that processes items from the queue"""
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            async with self._lock:
                self._active_tasks += 1

            try:
                self._logger.debug(f"{worker_name} processing task {item.id}")
                result = await item.coroutine_factory()
                item.future.set_result(result)
                self._total_processed += 1
            except Exception as e:
                self._total_errors += 1
                if not item.future.done():
                    item.future.set_exception(e)
                self._logger.error(f"{worker_name} error on task {item.id}: {e}")
            finally:
                async with self._lock:
                    self._active_tasks -= 1
                self._queue.task_done()

    async def submit(self, task_id: str, coroutine_factory: Callable[[], Coroutine]) -> Any:
        """
        Submit a task to the queue and wait for its result.

        Args:
            task_id: Unique identifier for the task
            coroutine_factory: Callable that returns a coroutine (not the coroutine itself)

        Returns:
            The result of the coroutine

        Raises:
            asyncio.QueueFull: If the queue is at capacity
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        item = QueueItem(
            id=task_id,
            coroutine_factory=coroutine_factory,
            future=future,
        )

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._logger.warning(f"Queue full, rejecting task {task_id}")
            raise

        self._logger.debug(
            f"Task {task_id} queued (position: {self._queue.qsize()}/{self._max_queue_size})"
        )

        return await future

    def get_stats(self) -> dict:
        """Get queue statistics"""
        return {
            "running": self._running,
            "workers": self._max_concurrent,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._max_queue_size,
            "active_tasks": self._active_tasks,
            "total_processed": self._total_processed,
            "total_errors": self._total_errors,
        }

    @property
    def is_full(self) -> bool:
        return self._queue.full()


# --- Singleton ---

_inference_queue: InferenceQueue | None = None


def get_inference_queue(max_concurrent: int = 2, max_queue_size: int = 50) -> InferenceQueue:
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = InferenceQueue(
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
        )
    return _inference_queue
