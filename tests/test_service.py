"""The macOS LaunchAgent commands, with launchctl replaced by a recorder. No test changes the real LaunchAgents."""

import plistlib
import subprocess

import pytest

from mlx_imagine import service


@pytest.fixture
def launchctl(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        assert command[0] == "launchctl", command
        calls.append(command[1:])
        return subprocess.CompletedProcess(command, 0, stdout="state = running\npid = 42\n", stderr="")

    monkeypatch.setattr(service.subprocess, "run", fake_run)
    monkeypatch.setattr(service.sys, "platform", "darwin")
    monkeypatch.setattr(service, "executable", lambda: "/opt/bin/mlx-imagine")
    monkeypatch.setattr(service, "_healthy", lambda url: True)
    return calls


def test_install_writes_a_plist_that_restarts_only_after_a_crash(launchctl, tmp_path) -> None:
    """Bug: KeepAlive is true without a throttle, so a server that fails at start restarts in a tight loop."""
    path = service.install(tmp_path / "config.toml", home=tmp_path)
    plist = plistlib.loads(path.read_bytes())
    assert plist["ProgramArguments"] == ["/opt/bin/mlx-imagine", "serve"]
    assert plist["KeepAlive"] == {"SuccessfulExit": False}
    assert plist["ThrottleInterval"] >= 10
    assert plist["EnvironmentVariables"]["MLX_IMAGINE_CONFIG"] == str(tmp_path / "config.toml")
    assert "api_token" not in path.read_text(errors="ignore")
    assert launchctl[-1][0] == "bootstrap"


def test_install_refuses_to_overwrite_without_force(launchctl, tmp_path) -> None:
    """Bug: a second install replaces a LaunchAgent the user edited, without a warning."""
    service.install(tmp_path / "config.toml", home=tmp_path)
    with pytest.raises(service.ServiceError, match="--force"):
        service.install(tmp_path / "config.toml", home=tmp_path)
    service.install(tmp_path / "config.toml", force=True, home=tmp_path)
    assert [call[0] for call in launchctl] == ["bootstrap", "bootout", "bootstrap"]


def test_uninstall_boots_out_and_deletes_the_plist(launchctl, tmp_path) -> None:
    """Bug: uninstall deletes the file but leaves the server running until the next login."""
    path = service.install(tmp_path / "config.toml", home=tmp_path)
    service.uninstall(home=tmp_path)
    assert not path.exists()
    assert launchctl[-1][0] == "bootout"
    with pytest.raises(service.ServiceError, match="not installed"):
        service.uninstall(home=tmp_path)


def test_status_reads_launchctl_and_health(launchctl, tmp_path) -> None:
    """Bug: status reports the server as running when only the plist exists."""
    status = service.status("http://127.0.0.1:8765/v8/health", home=tmp_path)
    assert status["state"] == "running" and status["pid"] == "42" and status["healthy"] is True


def test_the_service_commands_refuse_linux(monkeypatch) -> None:
    """Bug: install on Linux writes a plist that nothing reads and reports success."""
    monkeypatch.setattr(service.sys, "platform", "linux")
    with pytest.raises(service.ServiceError, match="macOS only"):
        service.require_macos()


def test_start_installs_once_stop_keeps_the_plist_and_restart_kicks(launchctl, tmp_path, monkeypatch) -> None:
    """Bug: stop deletes the LaunchAgent, or restart starts a second copy of the server."""
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(service, "_loaded", lambda: not any(call[0] == "bootout" for call in launchctl))
    config = tmp_path / "config.toml"
    assert service.start(config, home=tmp_path) == "installed and started"
    assert service.start(config, home=tmp_path) == "already running"
    assert service.restart(config, home=tmp_path) == "restarted"
    assert launchctl[-1][:2] == ["kickstart", "-k"]
    assert service.stop(home=tmp_path) == "stopped"
    assert "bootout" in [call[0] for call in launchctl]
    assert service.plist_path(tmp_path).exists()
    with pytest.raises(service.ServiceError, match="not installed"):
        service.stop(home=tmp_path / "other")


def test_stop_reports_a_server_that_does_not_exit(launchctl, tmp_path, monkeypatch) -> None:
    """Bug: stop says "stopped" while launchd still holds the job, and a start right after it does nothing."""
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)
    service.install(tmp_path / "config.toml", home=tmp_path)
    with pytest.raises(service.ServiceError, match="did not stop"):
        service.stop(home=tmp_path)  # the fake `print` always says the job is loaded


def test_status_shows_the_server_that_holds_the_lock(launchctl, tmp_path) -> None:
    """Bug: status shows no PID for a server that was not started by the LaunchAgent, so a user cannot find the
    process that blocks a second start."""
    from mlx_imagine.lock import ServerLock, lock_path

    assert service.status("http://127.0.0.1:1/v8/health", home=tmp_path)["lock_owner"] is None
    held = ServerLock(lock_path(tmp_path))
    held.acquire("http://127.0.0.1:8765")
    owner = service.status("http://127.0.0.1:1/v8/health", home=tmp_path)["lock_owner"]
    held.release()
    assert owner["address"] == "http://127.0.0.1:8765" and isinstance(owner["pid"], int)
