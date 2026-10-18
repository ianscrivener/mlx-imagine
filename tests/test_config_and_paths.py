"""Settings, the config file and the path checks."""

import os
import stat

import pytest

from mlx_imagine import config
from mlx_imagine.errors import ApiError
from mlx_imagine.paths import resolve_inside


def test_the_config_file_round_trips_and_is_private(tmp_path) -> None:
    """Bug: the config file holds the API token but other users can read it."""
    path = tmp_path / "config.toml"
    settings = config.Settings(api_token="secret", port=9000)
    config.save(settings, path)
    assert config.load(path) == settings
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_a_yaml_config_file_round_trips(tmp_path) -> None:
    """Bug: a .yaml config file is parsed as TOML, so the server stops at start with a parse error."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "# local test\n"
        "server:\n  port: 9001\n"
        "mflux_models:\n  local_directory: /tmp/models\n  some_future_key: 1\n"
        "mflux_config:\n  allowed_quantize: q4 q8 bf16\n"
        "memory_limit_gb: 28\n"
    )
    settings = config.load(path)
    assert (settings.port, settings.model_dir, settings.memory_limit_gb) == (9001, "/tmp/models", 28)
    assert settings.quantize_choices() == {4, 8, None}
    settings.port = 9002
    config.save(settings, path)
    assert config.load(path).port == 9002
    assert "some_future_key: 1" in path.read_text()  # an unknown key survives a save


def test_a_typo_in_allowed_quantize_is_refused_with_its_name(tmp_path) -> None:
    """Bug: an allowed_quantize token like qa5 is dropped without a word, so q5 loads fail with no reason."""
    path = tmp_path / "config.yaml"
    path.write_text("mflux_config:\n  allowed_quantize: q3 qa5\n")
    with pytest.raises(ApiError, match="qa5"):
        config.load(path)


def test_the_api_token_is_checked_only_when_required() -> None:
    """Bug: the placeholder token in a config with require_api_token: false locks every client out."""
    assert config.Settings(api_token="placeholder").token is None
    assert config.Settings(api_token="secret", require_api_token=True).token == "secret"
    with pytest.raises(ApiError, match="0.0.0.0"):
        config.check_startup(config.Settings(host="0.0.0.0"))
    with pytest.raises(ApiError, match="use_tls"):
        config.check_startup(config.Settings(use_tls=True))


def test_a_relative_directory_is_relative_to_the_config_file(tmp_path, monkeypatch) -> None:
    """Bug: output_dir: ./output resolves against the start directory, so the LaunchAgent writes images to /."""
    (tmp_path / "project").mkdir()
    path = tmp_path / "project" / "config.yaml"
    path.write_text("image_directories:\n  output: ./output\n")
    monkeypatch.chdir("/")
    assert config.load(path).path("output_dir") == (tmp_path / "project" / "output").resolve()


def test_a_bad_value_changes_nothing() -> None:
    """Bug: update applies the good values of a request before it refuses a bad one."""
    settings = config.Settings()
    with pytest.raises(ApiError):
        config.update(settings, {"queue_max_length": 5, "default_quantize": 7})
    assert settings.queue_max_length == 100


def test_apply_environment_points_mflux_and_the_hub_at_the_configured_directories(monkeypatch, tmp_path) -> None:
    """Bug: a user's HF_HUB_CACHE stays set and overrides huggingface_models.local_directory."""
    monkeypatch.setenv("HF_HUB_CACHE", "/somewhere/else")
    monkeypatch.setenv("MFLUX_CACHE_DIR", "/old")
    config.apply_environment(config.Settings(hf_hub_cache=str(tmp_path / "hub"), mflux_cache_dir=str(tmp_path / "mf")))
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hub")
    assert os.environ["MFLUX_CACHE_DIR"] == str(tmp_path / "mf")


def test_resolve_inside_refuses_traversal_and_symlinks_out(tmp_path) -> None:
    """Bug: '../' or a symlink inside the input directory reads a file outside it."""
    base = tmp_path / "input"
    base.mkdir()
    (tmp_path / "secret.txt").write_text("x")
    (base / "link.png").symlink_to(tmp_path / "secret.txt")
    assert resolve_inside(base, "a.png") == (base / "a.png").resolve()
    for candidate in ("../secret.txt", str(tmp_path / "secret.txt"), "link.png"):
        with pytest.raises(ApiError) as caught:
            resolve_inside(base, candidate)
        assert caught.value.code == "path_not_allowed"


def test_save_changes_writes_only_the_keys_the_client_sent(tmp_path) -> None:
    """Bug: a settings change writes every live setting to the file, so a `serve --host 0.0.0.0` override, the
    defaults and the absolute form of relative directories stay in the config file for each later start."""
    path = tmp_path / "config.yaml"
    path.write_text("image_directories:\n  output: ./output\nfuture_key: 1\n")
    settings = config.load(path)
    settings.host = "0.0.0.0"  # a serve --host override for this run only
    changes = {"queue": {"queue_max_length": 5}}
    config.update(settings, changes)
    config.save_changes(changes, path)
    written = path.read_text()
    assert "queue_max_length: 5" in written
    assert "output: ./output" in written  # still relative
    assert "0.0.0.0" not in written and "port" not in written
    assert "future_key: 1" in written


def test_the_startup_log_names_the_services_and_warns_when_no_token_is_on(caplog) -> None:
    """Bug: the server starts with no token and says nothing, and the user thinks the API is protected."""
    from mlx_imagine.cli import _log_startup

    _log_startup(config.Settings(swagger_enable=False))
    assert "REST API http://127.0.0.1:8765/v8" in caplog.text and "Swagger" not in caplog.text
    assert "No API token" in caplog.text
    caplog.clear()
    _log_startup(config.Settings(require_api_token=True, api_token="x"))
    assert "No API token" not in caplog.text
