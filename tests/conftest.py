"""Shared fixtures. The first statements make the whole session offline before anything imports mflux or
huggingface_hub (both read their cache paths once, at import): no test downloads a model, reads the user's model
cache or loads weights."""

import atexit
import json
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

from mlx_imagine import engine as engine_module, hf_models, saved_models
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
    # The Hugging Face cache is one folder for the session: empty it, so no test sees a model of another.
    shutil.rmtree(os.environ["HF_HUB_CACHE"], ignore_errors=True)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        lora_dir=str(tmp_path / "loras"),
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        data_dir=str(tmp_path / "data"),
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


def cache_repo(repo_id: str, mflux: bool = True, with_sidecar: bool = False, size_gb: float = 0) -> Path:
    """A repository in the Hugging Face cache of the session, with an mflux layout or a plain one. Returns the
    snapshot folder. `size_gb` adds a sparse weight file of that size, which takes no disk space."""
    folder = hf_models.repo_folder(repo_id)
    snapshot = folder / "snapshots" / ("a" * 40)
    (snapshot / "transformer").mkdir(parents=True)
    metadata = {"quantization_level": "8", "mflux_version": "0.18"} if mflux else {"total_size": 1}
    (snapshot / "transformer" / saved_models.INDEX).write_text(json.dumps({"metadata": metadata}))
    (folder / "refs").mkdir(exist_ok=True)
    (folder / "refs" / "main").write_text("a" * 40)
    if with_sidecar:
        saved_models.write_sidecar(snapshot, "z-image-turbo", "z-image-turbo", 8)
    if size_gb:
        with open(snapshot / "transformer" / "weights.safetensors", "wb") as file:
            file.truncate(int(size_gb * 1000**3))
    return snapshot
