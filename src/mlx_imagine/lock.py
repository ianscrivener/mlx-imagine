"""The single-server lock. One server owns the model and the GPU, so a second server must not start.

The lock is an flock on a file in the state directory. The kernel drops the lock when the process ends, even after
a crash, so a stale file never blocks a start. The file holds the PID and the address of the owner."""

import fcntl
import json
import os
from pathlib import Path
from typing import Any


class ServerRunning(Exception):
    """Another server holds the lock. pid and address come from the lock file and can be None."""

    def __init__(self, pid: int | None, address: str | None) -> None:
        self.pid = pid
        self.address = address
        super().__init__(f"A server already runs (PID {pid or 'unknown'}, {address or 'address unknown'}).")


def state_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "Application Support" / "mlx-imagine"


def lock_path(home: Path | None = None) -> Path:
    return state_dir(home) / "server.lock"


class ServerLock:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or lock_path()
        self._fd: int | None = None

    def acquire(self, address: str) -> None:
        """Take the lock, or raise ServerRunning. The lock stays until release or until the process ends."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pid, owner_address = _read_owner(fd)
            os.close(fd)
            raise ServerRunning(pid, owner_address) from None
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "address": address}).encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor drops the flock
            self._fd = None


def owner(path: Path | None = None) -> dict[str, Any] | None:
    """{"pid", "address"} of the server that holds the lock, or None when no server runs."""
    path = path or lock_path()
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            pid, address = _read_owner(fd)  # the lock is taken: a server runs
            return {"pid": pid, "address": address}
        return None
    finally:
        os.close(fd)


def running_address(path: Path | None = None) -> str | None:
    """The address of the server that holds the lock, or None when no server runs."""
    found = owner(path)
    return found["address"] if found else None


def _read_owner(fd: int) -> tuple[int | None, str | None]:
    try:
        data = json.loads(os.pread(fd, 4096, 0) or b"{}")
        pid, address = data.get("pid"), data.get("address")
        return (pid if isinstance(pid, int) else None), (address if isinstance(address, str) else None)
    except (OSError, ValueError, AttributeError):
        return None, None
