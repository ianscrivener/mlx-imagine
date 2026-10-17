"""The job queue: order, cancel, failure, limits."""

import threading

import pytest
from conftest import wait_for

from mlx_imagine.errors import ApiError
from mlx_imagine.jobs import Job, JobQueue


@pytest.fixture
def queue() -> JobQueue:
    jobs = JobQueue(max_length=3, history=10)
    jobs.start()
    yield jobs
    jobs.stop(timeout=5)


def test_jobs_run_one_at_a_time_in_order(queue: JobQueue) -> None:
    """Bug: two jobs run at the same time on two threads, or a later job runs first."""
    order, running = [], []

    def handler(name: str):
        def run(job: Job) -> str:
            running.append(name)
            assert len(running) == 1
            order.append(name)
            running.remove(name)
            return name

        return run

    jobs = [queue.submit(Job("generate", {}, handler(name))) for name in "abc"]
    for job in jobs:
        wait_for(job)
    assert order == ["a", "b", "c"]
    assert [job.result for job in jobs] == ["a", "b", "c"]


def test_a_waiting_job_cancels_at_once_and_a_running_job_at_its_next_check(queue: JobQueue) -> None:
    """Bug: a cancelled job that waits still runs, or a running job ignores the cancel."""
    started, release = threading.Event(), threading.Event()

    def slow(job: Job) -> None:
        started.set()
        release.wait(5)
        job.check_cancel()

    running = queue.submit(Job("generate", {}, slow))
    waiting = queue.submit(Job("generate", {}, lambda job: "ran"))
    started.wait(5)
    assert queue.position(waiting) == 1
    queue.cancel(waiting.id)
    assert waiting.state == "cancelled" and waiting.result is None
    queue.cancel(running.id)
    release.set()
    wait_for(running)
    assert running.state == "cancelled"
    assert running.error["code"] == "cancelled"
    with pytest.raises(ApiError) as caught:
        queue.cancel(running.id)
    assert caught.value.code == "job_ended"


def test_a_failed_job_does_not_stop_the_worker(queue: JobQueue) -> None:
    """Bug: an exception in one handler kills the worker thread, and every later job waits for ever."""

    def boom(job: Job) -> None:
        raise RuntimeError("mlx out of memory")

    failed = wait_for(queue.submit(Job("generate", {}, boom)))
    after = wait_for(queue.submit(Job("generate", {}, lambda job: "ok")))
    assert failed.state == "failed"
    assert failed.error["code"] == "generation_failed"
    assert failed.error["message"] == "mlx out of memory"
    assert after.state == "succeeded"


def test_a_full_queue_refuses_a_new_job() -> None:
    """Bug: the queue grows without a limit, so a client can fill the memory with jobs."""
    jobs = JobQueue(max_length=1)
    jobs.submit(Job("generate", {}, lambda job: None))
    with pytest.raises(ApiError) as caught:
        jobs.submit(Job("generate", {}, lambda job: None))
    assert (caught.value.code, caught.value.status) == ("queue_full", 429)


def test_the_job_object_never_shows_server_side_context(queue: JobQueue) -> None:
    """Bug: the job object carries the handler or the load spec, so the JSON response fails or leaks state."""
    job = Job("load", {"model": "z-image-turbo"}, lambda job: None)
    job.context["load_spec"] = object()
    described = queue.describe(job)
    assert set(described) == {
        "id", "type", "state", "position", "created_at", "started_at", "finished_at", "progress", "request",
        "result", "error", "warnings",
    }  # fmt: skip
