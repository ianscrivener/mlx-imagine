"""Server-Sent Events reach a real client one at a time while the job runs (ROADMAP T-06). The server runs under
uvicorn on a local port, so the test goes through a real socket and the real ASGI middleware."""

import http.client
import json
import socket
import threading
import time

import uvicorn
from _fake_command import FakeModel
from conftest import wait_for

from mlx_imagine.api import create_app
from mlx_imagine.engine import Engine


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_first_event_arrives_while_the_job_still_runs_and_the_token_is_checked(engine: Engine) -> None:
    """Bug: the auth middleware buffers the stream, and the client sees no event until the job ends."""
    engine.settings.require_api_token = True
    engine.settings.api_token = "secret"
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, lifespan="off", log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    opened: list[http.client.HTTPConnection] = []
    try:
        wait_for(engine.submit_load({"model": "z-image-turbo"}))
        FakeModel.hold_at = (0, 0)
        job = engine.submit_generate({"prompt": "a cat", "steps": 2, "seed": [0]})
        assert FakeModel.held.wait(10)
        refused = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        refused.request("GET", f"/v8/jobs/{job.id}/events")
        refused_response = refused.getresponse()
        assert refused_response.status == 401
        refused_response.read()
        refused.close()
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        opened.append(connection)
        connection.request("GET", f"/v8/jobs/{job.id}/events", headers={"authorization": "Bearer secret"})
        response = connection.getresponse()
        first = response.readline().decode()
        assert response.status == 200 and first.startswith("event:")  # the job is still held on step 0
        assert job.state == "running"
        json.loads(response.readline().decode().removeprefix("data: "))
    finally:
        FakeModel.release.set()
        for connection in opened:
            connection.close()
        server.should_exit = True
        thread.join(timeout=5)
