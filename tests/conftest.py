"""Shared fixtures. The first statements make the whole session offline before anything imports mflux or
huggingface_hub (both read their cache paths once, at import): no test downloads a model, reads the user's model
cache or loads weights."""

import atexit
import os
import shutil
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="mlx-imagine-hf-home-")
os.environ["HF_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["HF_XET_CACHE"] = os.path.join(os.environ["HF_HOME"], "xet")
os.environ["MFLUX_CACHE_DIR"] = os.path.join(os.environ["HF_HOME"], "mflux")
os.environ.pop("MLX_IMAGINE_CONFIG", None)
atexit.register(shutil.rmtree, os.environ["HF_HOME"], ignore_errors=True)

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import huggingface_hub.constants
import pytest
from _fake_command import FAKE_SPEC, FakeModel
from mflux.models.z_image.variants.z_image import ZImage

from mlx_imagine import engine as engine_module, hf_models
from mlx_imagine.config import Settings
from mlx_imagine.engine import Engine
from mlx_imagine.registry import Registry

huggingface_hub.constants.HF_HUB_OFFLINE = True
assert huggingface_hub.constants.is_offline_mode(), "the test session must run with the Hugging Face Hub offline"


def _no_real_model(self: Any, *args: Any, **kwargs: Any) -> None:
    raise AssertionError("a test constructed the real ZImage, which would load weights; use FakeModel")


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ZImage, "__init__", _no_real_model)
    FakeModel.hold_at = None
    FakeModel.held.clear()
    FakeModel.release.clear()
    FakeModel.fail_on_seed.clear()
    monkeypatch.setenv("MLX_IMAGINE_CONFIG", str(tmp_path / "config" / "config.toml"))
    FakeModel.instances.clear()
    yield
    FakeModel.instances.clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=str(tmp_path / "models"),
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        data_dir=str(tmp_path / "data"),
        mflux_cron_minutes=0,
        hf_cron_minutes=0,
    )


@pytest.fixture
def fake_registry() -> Registry:
    """A registry whose only command is the fake z-image-turbo command of tests/_fake_command.py."""
    return Registry(commands=(FAKE_SPEC,), stage_2=())


@pytest.fixture
def engine(settings: Settings, fake_registry: Registry, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    # The fake model is "in the cache": the real check would look for z-image-turbo's files.
    monkeypatch.setattr(hf_models, "is_complete", lambda entry: True)
    monkeypatch.setattr(engine_module.Engine, "_estimate_bytes", lambda self, spec: None)
    server = Engine(settings, fake_registry)
    server.start()
    yield server
    server.stop()


def wait_for(job: Any, timeout: float = 10.0) -> Any:
    """Block until the job ends."""
    subscriber = job.subscribe()
    try:
        while job.state not in {"succeeded", "failed", "cancelled"}:
            subscriber.get(timeout=timeout)
    finally:
        job.unsubscribe(subscriber)
    return job
