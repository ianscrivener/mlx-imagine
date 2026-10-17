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
    status = service.status("http://127.0.0.1:8765/v1/health", home=tmp_path)
    assert status["state"] == "running" and status["pid"] == "42" and status["healthy"] is True


def test_the_service_commands_refuse_linux(monkeypatch) -> None:
    """Bug: install on Linux writes a plist that nothing reads and reports success."""
    monkeypatch.setattr(service.sys, "platform", "linux")
    with pytest.raises(service.ServiceError, match="macOS only"):
        service.require_macos()
