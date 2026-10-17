"""Install the server as a macOS LaunchAgent (docs/scope.md 3.17). macOS only.

install writes ~/Library/LaunchAgents/<LABEL>.plist and loads it with launchctl bootstrap. uninstall unloads it
with launchctl bootout and deletes the file. status reads launchctl print and the server health endpoint."""

import os
import plistlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

LABEL = "io.github.ianscrivener.mlx-imagine"
# launchd waits this many seconds before it starts a server that exited again. Without it, a server that
# fails at start (for example, out of memory) starts again and again.
THROTTLE_SECONDS = 30


class ServiceError(Exception):
    pass


def require_macos() -> None:
    if sys.platform != "darwin":
        raise ServiceError("The service commands work on macOS only.")


def plist_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_dir(home: Path | None = None) -> Path:
    return (home or Path.home()) / "Library" / "Logs" / "mlx-imagine"


def executable() -> str:
    """The absolute path of the mlx-imagine command in this environment."""
    candidate = Path(sys.argv[0])
    if candidate.name == "mlx-imagine" and candidate.is_file():
        return str(candidate.resolve())
    found = shutil.which("mlx-imagine")
    if found is None:
        raise ServiceError("Cannot find the mlx-imagine command. Install the package, then run this again.")
    return str(Path(found).resolve())


def build_plist(program: str, config_file: Path, logs: Path) -> dict[str, Any]:
    return {
        "Label": LABEL,
        "ProgramArguments": [program, "serve"],
        # The config file holds the address and the token. The property list holds no secret.
        "EnvironmentVariables": {"MLX_IMAGINE_CONFIG": str(config_file), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": THROTTLE_SECONDS,
        "ProcessType": "Interactive",
        "StandardOutPath": str(logs / "server.log"),
        "StandardErrorPath": str(logs / "server.err.log"),
    }


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise ServiceError(f"launchctl {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def install(config_file: Path, force: bool = False, home: Path | None = None) -> Path:
    require_macos()
    path = plist_path(home)
    if path.exists():
        if not force:
            raise ServiceError(f"The LaunchAgent exists: {path}. Give --force to install it again.")
        _launchctl("bootout", f"{_domain()}/{LABEL}", check=False)
    logs = log_dir(home)
    logs.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        plistlib.dump(build_plist(executable(), config_file, logs), file)
    _launchctl("bootstrap", _domain(), str(path))
    return path


def uninstall(home: Path | None = None) -> Path:
    require_macos()
    path = plist_path(home)
    if not path.exists():
        raise ServiceError(f"The LaunchAgent is not installed: {path} does not exist.")
    _launchctl("bootout", f"{_domain()}/{LABEL}", check=False)
    path.unlink()
    return path


def status(health_url: str, home: Path | None = None) -> dict[str, Any]:
    require_macos()
    path = plist_path(home)
    printed = _launchctl("print", f"{_domain()}/{LABEL}", check=False)
    state = None
    pid = None
    for line in printed.stdout.splitlines():
        line = line.strip()
        if line.startswith("state = "):
            state = line.split("=", 1)[1].strip()
        elif line.startswith("pid = "):
            pid = line.split("=", 1)[1].strip()
    return {
        "installed": path.exists(),
        "plist": str(path),
        "loaded": printed.returncode == 0,
        "state": state,
        "pid": pid,
        "healthy": _healthy(health_url),
        "logs": str(log_dir(home)),
    }


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 -- a local http URL we build
            return response.status == 200
    except OSError:
        return False
