"""The job queue: one worker thread runs one job at a time, in the order the jobs arrive (docs/scope.md 3.7).

All MLX work happens on the worker thread. The HTTP and MCP threads only add jobs, read them and cancel them."""

import itertools
import logging
import os
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import ApiError, conflict, not_found

logger = logging.getLogger(__name__)

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
ENDED = frozenset({SUCCEEDED, FAILED, CANCELLED})


class JobCancelled(Exception):
    """Raised inside a job handler when a client cancelled the job."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


_counter = itertools.count(1)


def _new_id() -> str:
    return f"job_{int(time.time()):x}{os.getpid() % 0xFFF:03x}{next(_counter):04x}"


@dataclass
class Job:
    type: str
    request: dict[str, Any]
    handler: Callable[["Job"], Any]
    id: str = field(default_factory=_new_id)
    state: str = QUEUED
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] | None = None
    result: Any = None
    error: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    # Server-side data the handler and the queue checks use. Never sent to a client.
    context: dict[str, Any] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _subscribers: list[queue.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()

    def emit(self, event: str, data: Any) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.put((event, data))

    def set_progress(self, progress: dict[str, Any]) -> None:
        self.progress = progress
        self.emit("progress", progress)

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        if self.state in ENDED:
            subscriber.put(("end", None))
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def to_dict(self, position: int | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "state": self.state,
            "position": position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "request": self.request,
            "result": self.result,
            "error": self.error,
            "warnings": self.warnings,
        }


class JobQueue:
    def __init__(self, max_length: int = 100, history: int = 200) -> None:
        self.max_length = max_length
        self._history: deque[Job] = deque(maxlen=history)
        self._waiting: deque[Job] = deque()
        self._jobs: dict[str, Job] = {}
        self._running: Job | None = None
        self._condition = threading.Condition()
        self._stop = False
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="mlx-imagine-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        with self._condition:
            self._stop = True
            for job in self._waiting:
                job.cancel_event.set()
            if self._running:
                self._running.cancel_event.set()
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def set_history(self, history: int) -> None:
        with self._condition:
            self._history = deque(self._history, maxlen=history)

    # -- client side -----------------------------------------------------------------------------------------

    def submit(self, job: Job, before_queue: Callable[[], None] | None = None) -> Job:
        """Add a job. before_queue runs under the queue lock, so a check and the enqueue are one step."""
        with self._condition:
            if len(self._waiting) >= self.max_length:
                raise ApiError("queue_full", 429, f"The queue has {self.max_length} jobs. Try again later.")
            if before_queue is not None:
                before_queue()
            self._waiting.append(job)
            self._jobs[job.id] = job
            self._condition.notify_all()
        return job

    def get(self, job_id: str) -> Job:
        with self._condition:
            job = self._jobs.get(job_id)
        if job is None:
            raise not_found(f"No job with ID {job_id!r}.", job=job_id)
        return job

    def position(self, job: Job) -> int | None:
        with self._condition:
            if job is self._running:
                return 0
            for index, waiting in enumerate(self._waiting):
                if waiting is job:
                    return index + 1
        return None

    def describe(self, job: Job) -> dict[str, Any]:
        return job.to_dict(self.position(job))

    def list_jobs(self, state: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._condition:
            active = ([self._running] if self._running else []) + list(self._waiting)
            ended = list(self._history)[-limit:][::-1]
        jobs = [job for job in active + ended if state is None or job.state == state]
        return [self.describe(job) for job in jobs]

    def with_waiting(self, fn: Callable[[list[Job]], None]) -> None:
        """Run fn with the waiting jobs, under the queue lock, so a check and a change are one step."""
        with self._condition:
            fn(list(self._waiting))

    def waiting(self) -> list[Job]:
        with self._condition:
            return list(self._waiting)

    @property
    def running(self) -> Job | None:
        return self._running

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._condition:
            if job.state in ENDED:
                raise conflict("job_ended", f"The job {job_id!r} has already ended.", job=job_id)
            job.cancel_event.set()
            if job in self._waiting:
                self._waiting.remove(job)
                self._finish(job, CANCELLED, error={"code": "cancelled", "message": "A client cancelled the job."})
        return job

    # -- worker side -----------------------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._waiting and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                job = self._waiting.popleft()
                self._running = job
                job.state = RUNNING
                job.started_at = _now()
            job.emit("state", self.describe(job))
            self._run(job)
            with self._condition:
                self._running = None

    def _run(self, job: Job) -> None:
        try:
            job.check_cancel()
            result = job.handler(job)
        except JobCancelled:
            self._finish(job, CANCELLED, error={"code": "cancelled", "message": "A client cancelled the job."})
        except ApiError as exc:
            self._finish(job, FAILED, error=exc.to_dict())
        except Exception as exc:  # noqa: BLE001 -- one failed job must not stop the worker
            logger.exception("job %s failed", job.id)
            code = {"generate": "generation_failed", "download": "download_failed"}.get(job.type, "job_failed")
            details = {"exception": type(exc).__name__, "traceback": traceback.format_exc(limit=8)}
            self._finish(
                job,
                CANCELLED if job.cancelled else FAILED,
                error={"code": code, "message": str(exc), "details": details},
            )
        else:
            self._finish(job, SUCCEEDED, result=result)

    def _finish(self, job: Job, state: str, result: Any = None, error: dict[str, Any] | None = None) -> None:
        if result is not None:
            job.result = result
        job.error = error
        job.state = state
        job.finished_at = _now()
        job.emit("state", job.to_dict(None))
        job.emit("end", job.to_dict(None))
        with self._condition:
            self._history.append(job)
            # Keep the job findable by ID only while it is in the history.
            live = {j.id for j in self._history} | {j.id for j in self._waiting}
            if self._running is not None:
                live.add(self._running.id)
            live.add(job.id)
            for stale in [job_id for job_id in self._jobs if job_id not in live]:
                del self._jobs[stale]
