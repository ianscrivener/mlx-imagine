"""LoRA files: list, name checks, apply at load, download and delete."""

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlx_imagine import loras
from mlx_imagine.api import create_app
from mlx_imagine.engine import Engine


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    with TestClient(create_app(engine)) as test_client:
        yield test_client


def _lora(engine: Engine, name: str, size: int = 10) -> Path:
    path = engine.lora_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not end")


def test_the_list_shows_safetensors_files_and_skips_other_files(engine: Engine, client: TestClient) -> None:
    """Bug: the list shows a hidden file, a file that is not a LoRA, or misses a file in a subfolder."""
    _lora(engine, "a.safetensors")
    _lora(engine, "style/b.safetensors")
    _lora(engine, "notes.txt")
    _lora(engine, ".hidden/c.safetensors")
    names = [item["name"] for item in client.get("/v1/loras").json()["loras"]]
    assert names == ["a.safetensors", "style/b.safetensors"]


@pytest.mark.parametrize("name", ["../x.safetensors", "/etc/x.safetensors", "a.txt", 5, "missing.safetensors"])
def test_a_name_outside_the_lora_directory_or_not_a_lora_is_refused(engine: Engine, name: object) -> None:
    """Bug: a client reads a file outside the LoRA directory, or a file that is not a LoRA."""
    with pytest.raises(Exception) as error:
        loras.resolve(engine.lora_dir, name)
    assert getattr(error.value, "status", None) in {400, 403, 404}


def test_a_load_with_loras_passes_the_paths_and_scales_to_the_command(engine: Engine, client: TestClient) -> None:
    """Bug: the loras of a load request do not reach the model, or the scale is lost."""
    path = _lora(engine, "style/b.safetensors")
    response = client.post(
        "/v1/model/load",
        json={
            "model": "z-image-turbo",
            "loras": ["style/b.safetensors", {"name": "style/b.safetensors", "scale": 0.5}],
        },
    )
    assert response.status_code == 202, response.text
    assert _wait(client, response.json()["id"])["state"] == "succeeded"
    from _fake_command import FakeModel

    args = FakeModel.instances[-1].args
    assert args.lora_paths == [str(path.resolve())] * 2 and args.lora_scales == [1.0, 0.5]
    assert client.get("/v1/model").json()["loras"] == [
        {"name": "style/b.safetensors", "scale": 1.0},
        {"name": "style/b.safetensors", "scale": 0.5},
    ]


@pytest.mark.parametrize(
    "loras_value",
    [[], "a.safetensors", [5], [{"name": "a.safetensors", "scale": "high"}], [{"scale": 1}], ["x.safetensors"] * 9],
)
def test_a_bad_loras_field_is_refused_before_the_job_enters_the_queue(
    engine: Engine, client: TestClient, loras_value: object
) -> None:
    """Bug: a bad loras field fails inside the queue, or loads a model without the LoRA."""
    _lora(engine, "a.safetensors")
    response = client.post("/v1/model/load", json={"model": "z-image-turbo", "loras": loras_value})
    assert response.status_code in {400, 404}, response.text


def test_the_lora_options_of_mflux_stay_blocked_and_point_to_loras(client: TestClient) -> None:
    """Bug: --lora reaches mflux, which resolves a Hugging Face name and downloads it."""
    response = client.post("/v1/model/load", json={"model": "z-image-turbo", "lora_paths": ["owner/name"]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "option_blocked" and "loras" in response.json()["error"]["message"]


def test_delete_needs_confirm_and_refuses_a_lora_in_use(engine: Engine, client: TestClient) -> None:
    """Bug: a delete without confirm removes the file, or removes a LoRA of the loaded model."""
    path = _lora(engine, "style/b.safetensors")
    assert client.delete("/v1/loras/style/b.safetensors").json()["error"]["code"] == "confirm_required"
    load = client.post("/v1/model/load", json={"model": "z-image-turbo", "loras": ["style/b.safetensors"]})
    _wait(client, load.json()["id"])
    in_use = client.delete("/v1/loras/style/b.safetensors", params={"confirm": True})
    assert in_use.status_code == 409 and in_use.json()["error"]["code"] == "lora_in_use"
    _wait(client, client.post("/v1/model/unload").json()["id"])
    done = client.delete("/v1/loras/style/b.safetensors", params={"confirm": True})
    assert done.status_code == 200 and not path.exists() and not path.parent.exists()


def test_a_download_gets_one_file_into_a_folder_of_the_repository(
    engine: Engine, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the download writes outside the LoRA directory, or fetches the whole repository."""
    calls: list[dict[str, Any]] = []

    def fake_snapshot(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["allow_patterns"][0]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")
        return str(target.parent)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot)
    response = client.post("/v1/loras/downloads", json={"repo_id": "owner/name", "filename": "sub/style.safetensors"})
    assert response.status_code == 202, response.text
    job = _wait(client, response.json()["id"])
    assert job["state"] == "succeeded", job["error"]
    assert job["result"]["name"] == "owner--name/sub/style.safetensors"
    assert calls[0]["allow_patterns"] == ["sub/style.safetensors"]
    assert [item["name"] for item in client.get("/v1/loras").json()["loras"]] == ["owner--name/sub/style.safetensors"]


@pytest.mark.parametrize(
    "body",
    [
        {"repo_id": "owner", "filename": "a.safetensors"},
        {"repo_id": "owner/name", "filename": "../a.safetensors"},
        {"repo_id": "owner/name", "filename": "a.bin"},
        {"repo_id": "owner/name", "filename": "/a.safetensors"},
        {"repo_id": "../x", "filename": "a.safetensors"},
    ],
)
def test_a_download_with_a_bad_repository_or_file_is_refused(client: TestClient, body: dict[str, str]) -> None:
    """Bug: a repository ID or file name with a path part writes outside the LoRA directory."""
    assert client.post("/v1/loras/downloads", json=body).status_code == 400


def test_a_download_needs_confirm_when_the_setting_asks_for_it(engine: Engine, client: TestClient) -> None:
    """Bug: require_model_confirmation does not apply to a LoRA download."""
    engine.settings.hf_require_model_confirmation = True
    response = client.post("/v1/loras/downloads", json={"repo_id": "owner/name", "filename": "a.safetensors"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "confirm_required"
