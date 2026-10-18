import json
import os
from pathlib import Path

import pytest

from mlx_imagine.lock import ServerLock, ServerRunning


def test_a_second_lock_fails_and_names_the_owner(tmp_path: Path) -> None:
    """Bug: two servers start and each loads a model."""
    path = tmp_path / "server.lock"
    first, second = ServerLock(path), ServerLock(path)
    first.acquire("http://127.0.0.1:8765")
    with pytest.raises(ServerRunning) as error:
        second.acquire("http://127.0.0.1:9999")
    assert error.value.pid == os.getpid()
    assert error.value.address == "http://127.0.0.1:8765"
    first.release()


def test_the_lock_is_free_after_release_and_a_stale_file_does_not_block(tmp_path: Path) -> None:
    """Bug: a file left by a crashed server blocks every later start."""
    path = tmp_path / "server.lock"
    path.write_text(json.dumps({"pid": 1, "address": "http://127.0.0.1:1"}))
    lock = ServerLock(path)
    lock.acquire("http://127.0.0.1:8765")
    assert json.loads(path.read_text())["pid"] == os.getpid()
    lock.release()
    ServerLock(path).acquire("http://127.0.0.1:8765")


def test_running_address_gives_the_address_of_the_server_that_holds_the_lock(tmp_path: Path) -> None:
    """Bug: the stdio client checks the config port, so it misses a server that started with --port."""
    from mlx_imagine.lock import running_address

    path = tmp_path / "server.lock"
    assert running_address(path) is None
    lock = ServerLock(path)
    lock.acquire("http://127.0.0.1:9999")
    assert running_address(path) == "http://127.0.0.1:9999"
    lock.release()
    assert running_address(path) is None
