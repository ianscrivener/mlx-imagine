"""The OpenAI-compatible routes over the fake command."""

import base64
import io

import PIL.Image
import pytest
from fastapi.testclient import TestClient

from mlx_imagine.api import create_app
from mlx_imagine.engine import Engine
from mlx_imagine.openai_api import build_request, parse_size


@pytest.fixture
def client(engine: Engine) -> TestClient:
    return TestClient(create_app(engine))


def test_size_and_n_become_a_native_request() -> None:
    """Bug: size 1024x512 swaps width and height, or n=3 gives one seed and so one image."""
    assert parse_size("1024x512") == {"width": 1024, "height": 512}
    assert parse_size("auto") == {}
    request = build_request({"prompt": "x", "n": 3, "size": "64x64"})
    assert (request["width"], request["height"], len(request["seed"])) == (64, 64, 3)
    assert build_request({"prompt": "x", "seed": 7})["seed"] == [7]


def test_generations_loads_the_model_and_returns_base64_images(client: TestClient) -> None:
    """Bug: an OpenAI client gets a job object, not images, or the model that it names never loads."""
    response = client.post(
        "/v1/images/generations",
        json={"model": "z-image-turbo", "prompt": "a cat", "n": 2, "size": "128x128", "steps": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["created"], int) and len(body["data"]) == 2
    image = PIL.Image.open(io.BytesIO(base64.b64decode(body["data"][0]["b64_json"])))
    assert image.format == "PNG"
    urls = client.post("/v1/images/generations", json={"prompt": "a cat", "response_format": "url", "steps": 1}).json()[
        "data"
    ]
    assert urls[0]["url"].startswith("http://testserver/v1/images/")
    assert client.get(urls[0]["url"].removeprefix("http://testserver")).status_code == 200


def test_bad_openai_requests_are_400(client: TestClient) -> None:
    """Bug: a bad size or n returns a 500, or starts a job."""
    for body in (
        {"prompt": "x", "size": "big"},
        {"prompt": "x", "n": 0},
        {"n": 1},
        {"prompt": "x", "response_format": "x"},
    ):
        assert client.post("/v1/images/generations", json={"model": "z-image-turbo", **body}).status_code == 400, body
    assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 400  # no model loaded


def test_models_lists_the_local_models_in_the_openai_shape(client: TestClient, engine: Engine) -> None:
    """Bug: GET /v1/models returns the native shape, and an OpenAI client cannot read the list."""
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and isinstance(body["data"], list)
    assert all(item["object"] == "model" and item["owned_by"] == "mlx-imagine" for item in body["data"])


def test_edits_takes_a_multipart_image(client: TestClient) -> None:
    """Bug: /v1/images/edits ignores the uploaded image, so the result does not start from it."""
    buffer = io.BytesIO()
    PIL.Image.new("RGB", (64, 64)).save(buffer, "PNG")
    response = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", buffer.getvalue(), "image/png")},
        data={"prompt": "a hat", "model": "z-image-turbo", "size": "64x64"},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == 1


def test_edits_accepts_an_upload_larger_than_the_json_limit(client: TestClient) -> None:
    """Bug: the body limit gives /v1/images/edits the 1 MB JSON limit, so a normal photo gets 413."""
    import os

    buffer = io.BytesIO()
    PIL.Image.frombytes("RGB", (700, 700), os.urandom(700 * 700 * 3)).save(buffer, "PNG")
    assert len(buffer.getvalue()) > 1024 * 1024
    response = client.post(
        "/v1/images/edits",
        files={"image": ("a.png", buffer.getvalue(), "image/png")},
        data={"prompt": "a hat", "model": "z-image-turbo", "size": "64x64"},
    )
    assert response.status_code == 200, response.text
