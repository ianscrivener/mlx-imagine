"""Per-request input and output subfolders."""

import io
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import PIL.Image
import pytest
from fastapi.testclient import TestClient

from mlx_imagine.api import create_app
from mlx_imagine.engine import Engine
from mlx_imagine.paths import subdir


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    with TestClient(create_app(engine)) as test_client:
        yield test_client


def _wait(client: TestClient, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/v8/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not end")


def _load(client: TestClient) -> None:
    _wait(client, client.put("/v8/engine/model", json={"model": "z-image-turbo"}).json()["id"])


def _png() -> bytes:
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (64, 64)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "value", ["", "/abs", "../x", "a/../b", "a//b", ".hidden", "a/b/c/d", "a\\b", 5, None, "a/./b"]
)
def test_a_subfolder_name_that_leaves_the_directory_is_refused(tmp_path: Path, value: object) -> None:
    """Bug: a request writes or reads outside the configured input or output directory."""
    with pytest.raises(Exception) as error:
        subdir(tmp_path, value)
    assert getattr(error.value, "status", None) in {400, 403}


def test_the_images_go_into_the_output_subfolder(engine: Engine, client: TestClient) -> None:
    """Bug: output_subdir is ignored, or the server makes the folder outside the output directory."""
    _load(client)
    response = client.post(
        "/v8/generate", json={"prompt": "a cat", "seed": [3], "steps": 1, "output_subdir": "cats/2026"}
    )
    assert response.status_code == 202, response.text
    job = _wait(client, response.json()["id"])
    assert job["state"] == "succeeded", job["error"]
    path = Path(job["result"]["images"][0]["path"])
    assert path.parent == (engine.output_dir / "cats" / "2026").resolve() and path.is_file()
    assert client.get(job["result"]["images"][0]["url"]).status_code == 200


@pytest.mark.parametrize("name", ["../out", "/tmp/x", ".hidden"])
def test_a_bad_output_subfolder_fails_before_the_job_enters_the_queue(client: TestClient, name: str) -> None:
    """Bug: a request that waits in the queue fails later for a name the server can see at once."""
    _load(client)
    response = client.post("/v8/generate", json={"prompt": "a cat", "output_subdir": name})
    assert response.status_code in {400, 403}, response.text
    assert client.get("/v8/jobs").json()["jobs"][0]["type"] == "load"


def test_an_upload_goes_into_the_input_subfolder_and_its_id_still_works(engine: Engine, client: TestClient) -> None:
    """Bug: the upload ID is not found when the upload is in a subfolder."""
    response = client.post("/v8/uploads", params={"subdir": "refs"}, files={"file": ("a.png", _png(), "image/png")})
    assert response.status_code == 201, response.text
    assert Path(response.json()["path"]).parent == (engine.input_dir / "refs").resolve()
    assert engine.resolve_input(response.json()["upload_id"]).is_file()


def test_input_subdir_is_the_base_of_relative_image_paths(engine: Engine, client: TestClient) -> None:
    """Bug: input_subdir is ignored, or a relative path leaves the subfolder."""
    (engine.input_dir / "refs").mkdir(parents=True)
    (engine.input_dir / "refs" / "a.png").write_bytes(_png())
    (engine.input_dir / "b.png").write_bytes(_png())
    assert engine.resolve_input("a.png", engine.input_dir / "refs") == (engine.input_dir / "refs" / "a.png").resolve()
    with pytest.raises(Exception) as error:
        engine.resolve_input("../b.png", engine.input_dir / "refs")
    assert getattr(error.value, "status", None) == 403
    _load(client)
    missing = client.post("/v8/generate", json={"prompt": "x", "input_subdir": "nope"})
    assert missing.status_code == 404
