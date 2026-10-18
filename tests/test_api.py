"""The REST API over the fake command: auth, error bodies, the job flow, uploads and config."""

import io
import os
import time
from collections.abc import Iterator

import PIL.Image
import pytest
from conftest import wait_for
from fastapi.testclient import TestClient

from mlx_imagine.api import create_app
from mlx_imagine.engine import Engine


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    with TestClient(create_app(engine)) as test_client:
        yield test_client


def _wait(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not end")


def test_load_generate_and_fetch_the_image(client: TestClient) -> None:
    """Bug: a route returns the wrong status, or the image URL in the result does not serve the file."""
    load = client.post("/v1/model/load", json={"model": "z-image-turbo"})
    assert load.status_code == 202
    generate = client.post("/v1/generate", json={"prompt": "a cat", "seed": [7], "steps": 1})
    assert generate.status_code == 202
    job = _wait(client, generate.json()["id"])
    assert job["state"] == "succeeded", job["error"]
    image = client.get(job["result"]["images"][0]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert client.get("/v1/model").json()["model_name"] == "z-image-turbo"


def test_an_error_has_the_documented_body(client: TestClient) -> None:
    """Bug: an ApiError reaches the client as a bare 500 without its code."""
    response = client.post("/v1/generate", json={"prompt": "a cat"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_loaded"
    response = client.get("/v1/jobs/job_missing")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


def test_the_token_is_required_on_every_route_but_health(engine: Engine) -> None:
    """Bug: a route skips the token check, or the health check needs a token the LaunchAgent status cannot give."""
    engine.settings.api_token = "secret"
    engine.settings.require_api_token = True
    with TestClient(create_app(engine)) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/server").status_code == 401
        assert client.get("/v1/server", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/server", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert (
            client.get("/v1/config", headers={"Authorization": "Bearer secret"}).json()["config"]["server"]["api_token"]
            == "***"
        )


def test_an_upload_can_be_the_image_path_of_a_generate_request(client: TestClient, engine: Engine) -> None:
    """Bug: an upload lands outside the input directory, or its ID does not resolve in a generate request."""
    data = io.BytesIO()
    PIL.Image.new("RGB", (64, 32)).save(data, format="PNG")
    upload = client.post("/v1/uploads", files={"file": ("x.png", data.getvalue(), "image/png")})
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert body["path"].startswith(str(engine.input_dir)) and (body["width"], body["height"]) == (64, 32)
    client.post("/v1/model/load", json={"model": "z-image-turbo"})
    job = client.post("/v1/generate", json={"prompt": "a cat", "image_path": body["upload_id"], "steps": 1}).json()
    assert _wait(client, job["id"])["state"] == "succeeded"
    not_an_image = client.post("/v1/uploads", files={"file": ("x.png", b"not a png", "image/png")})
    assert not_an_image.status_code == 400


def test_a_config_change_reports_which_settings_need_a_restart(client: TestClient, tmp_path) -> None:
    """Bug: a change to the hub directory claims to apply at once, though huggingface_hub read the old value at
    import."""
    changes = {"queue": {"queue_max_length": 5}, "huggingface_models": {"local_directory": str(tmp_path / "hub")}}
    response = client.patch("/v1/config", json=changes)
    assert response.status_code == 200, response.text
    assert response.json()["applied"] == ["queue.queue_max_length"]
    assert response.json()["restart_required"] == ["huggingface_models.local_directory"]
    assert client.get("/v1/server").json()["restart_required"] is True
    assert client.patch("/v1/config", json={"server": {"port": "x"}}).status_code == 400
    assert client.patch("/v1/config", json={"no_such": 1}).status_code == 400


def test_the_allow_flags_and_limits_refuse_downloads_and_deletes(client: TestClient, engine: Engine) -> None:
    """Bug: allow_delete: false in the config does not stop a delete request."""
    engine.settings.hf_allow_delete = False
    engine.settings.mflux_allow_delete = False
    engine.settings.hf_allow_access = False
    assert client.delete("/v1/models/mflux/mflux_dir/x?confirm=true").json()["error"]["code"] == "operation_disabled"
    assert client.delete("/v1/models/mflux/hf_cache/a/b?confirm=true").status_code == 403
    response = client.post("/v1/models/mflux/downloads", json={"model": "z-image-turbo"})
    assert response.status_code == 403
    engine.settings.mflux_max_models = 1
    (engine.models_dir / "one" / "vae").mkdir(parents=True)
    (engine.models_dir / "one" / "vae" / "model.safetensors.index.json").write_text("{}")
    response = client.post("/v1/models/mflux/saves", json={"model": "z-image-turbo", "quantize": 8})
    assert response.json()["error"]["code"] == "limit_reached"


def test_a_quantization_outside_allowed_quantize_is_refused(client: TestClient, engine: Engine) -> None:
    """Bug: allowed_quantize lists q8 only, but a load with q4 or without quantization still starts."""
    engine.settings.allowed_quantize = "q8"
    assert client.post("/v1/model/load", json={"model": "z-image-turbo", "quantize": 4}).status_code == 422
    assert client.post("/v1/model/load", json={"model": "z-image-turbo"}).status_code == 422
    assert client.post("/v1/model/load", json={"model": "z-image-turbo", "quantize": 8}).status_code == 202


def test_the_supported_list_and_the_schema_route(client: TestClient) -> None:
    """Bug: the schema route serves a model the server cannot run, or omits the generate schema."""
    models = {model["name"]: model for model in client.get("/v1/models/mflux/registry").json()["models"]}
    assert models["z-image-turbo"]["supported"] is True
    assert models["dev"]["supported"] is False
    schema = client.get("/v1/models/mflux/schemas/z-image-turbo").json()
    assert schema["generate_schema"]["required"] == ["prompt"]
    assert client.get("/v1/models/mflux/schemas/dev").status_code == 422


def test_the_event_stream_ends_with_the_final_job(client: TestClient) -> None:
    """Bug: the stream never closes after the job ends, so a client hangs."""
    client.post("/v1/model/load", json={"model": "z-image-turbo"})
    job = client.post("/v1/generate", json={"prompt": "a cat", "steps": 2}).json()
    with client.stream("GET", f"/v1/jobs/{job['id']}/events") as stream:
        text = "".join(stream.iter_text())
    assert text.startswith("event: state")
    assert "event: end" in text and '"state": "succeeded"' in text


@pytest.mark.parametrize("name", [123, ["a"], None])
def test_a_save_name_that_is_not_a_string_is_a_400_not_a_500(client: TestClient, name) -> None:
    """Bug: a numeric save name reaches safe_name's `"/" in name`, raises TypeError and returns a bare 500."""
    response = client.post("/v1/models/mflux/saves", json={"model": "z-image-turbo", "quantize": 8, "name": name})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_download_asks_for_confirmation_with_the_size(client: TestClient, engine: Engine, monkeypatch) -> None:
    """Bug: require_model_confirmation is true, and a download request without confirm starts a download of
    gigabytes with no question."""
    engine.settings.hf_require_model_confirmation = True
    monkeypatch.setattr("mlx_imagine.hf_models.estimate_bytes", lambda entry: 4_200_000_000)
    response = client.post("/v1/models/mflux/downloads", json={"model": "z-image-turbo"})
    assert response.status_code == 409 and response.json()["error"]["code"] == "confirm_required"
    assert "4.2 GB" in response.json()["error"]["message"]
    engine.settings.hf_allow_model_download = False
    assert client.post("/v1/models/mflux/downloads", json={"model": "z-image-turbo"}).status_code == 202


def test_offline_mode_follows_allow_hf_access(monkeypatch) -> None:
    """Bug: allow_hf_access is false, and huggingface_hub still reaches the network."""
    from mlx_imagine.config import Settings, apply_environment

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    apply_environment(Settings())
    assert "HF_HUB_OFFLINE" not in os.environ
    apply_environment(Settings(hf_allow_access=False))
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_the_swagger_page_follows_the_services_settings_and_needs_no_token(client: TestClient, engine: Engine) -> None:
    """Bug: /docs answers 401 when a token is on, or stays up when swagger_enable is false."""
    engine.settings.require_api_token = True
    engine.settings.api_token = "secret"
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert client.get("/v1/server").status_code == 401
    engine.settings.api_enable = False
    assert (
        client.get("/v1/server", headers={"authorization": "Bearer secret"}).status_code == 200
    )  # a restart applies it
    off = TestClient(create_app(engine))
    response = off.get("/v1/server", headers={"authorization": "Bearer secret"})
    assert response.status_code == 404 and response.json()["error"]["code"] == "api_disabled"
    assert off.get("/v1/health").status_code == 200


def test_swagger_enable_false_removes_the_docs_routes(engine: Engine) -> None:
    """Bug: swagger_enable is false, and /docs or /openapi.json still answers."""
    from mlx_imagine.api import create_app

    engine.settings.swagger_enable = False
    off = TestClient(create_app(engine))
    assert off.get("/docs").status_code == 404 and off.get("/openapi.json").status_code == 404


def test_the_swagger_schema_offers_select_lists_and_body_examples(client: TestClient) -> None:
    """Bug: the Swagger page shows a free text box for location or store, and an empty body for a POST."""
    schema = client.get("/openapi.json").json()
    params = {p["name"]: p for p in schema["paths"]["/v1/models/mflux"]["get"]["parameters"]}

    def options(parameter: dict) -> list[str]:
        node = parameter["schema"]
        node = next(item for item in node.get("anyOf", [node]) if "enum" in item)
        return node["enum"]

    assert options(params["location"]) == ["local", "remote"]
    assert options(params["format"]) == ["hf", "mflux"]
    assert options(params["store"]) == ["hf_cache", "mflux_dir", "mflux_community", "hub"]
    body = schema["paths"]["/v1/generate"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body["examples"][0]["prompt"]
    assert client.get("/v1/models/mflux", params={"store": "nope"}).status_code == 400


def test_a_body_over_the_limit_is_refused_before_the_app_reads_it(client: TestClient) -> None:
    """Bug: a 100 MB JSON body is read into memory, because nothing limits the request body."""
    big = "x" * (2 * 1024 * 1024)
    response = client.post("/v1/generate", content=b'{"prompt": "' + big.encode() + b'"}',
                           headers={"content-type": "application/json"})  # fmt: skip
    assert response.status_code == 413 and response.json()["error"]["code"] == "body_too_large"

    def chunks():
        for _ in range(4):
            yield b"x" * (512 * 1024)

    streamed = client.post("/v1/generate", content=chunks(), headers={"content-type": "application/json"})
    assert streamed.status_code == 413  # no Content-Length header: the count of the bytes stops it


def test_a_generate_request_over_the_size_limits_is_refused_before_the_queue(
    client: TestClient, engine: Engine
) -> None:
    """Bug: 1000 seeds or 8192x8192 enters the queue and uses all the memory."""
    wait_for(engine.submit_load({"model": "z-image-turbo"}))
    cases = [{"seed": list(range(engine.settings.max_seeds + 1))}, {"width": 8192, "height": 8192},
             {"steps": engine.settings.max_steps + 1}]  # fmt: skip
    for extra in cases:
        response = client.post("/v1/generate", json={"prompt": "x", **extra})
        assert response.status_code == 422 and response.json()["error"]["code"] == "limit_exceeded", extra
    assert engine.queue.waiting() == []


def test_an_upload_reports_the_size_after_the_exif_turn(client: TestClient) -> None:
    """Bug: a phone photo with orientation 6 reports its width and height the wrong way round, so the client
    sets a wrong output size."""
    import io

    import piexif
    import PIL.Image

    buffer = io.BytesIO()
    PIL.Image.new("RGB", (200, 100)).save(buffer, "JPEG", exif=piexif.dump({"0th": {piexif.ImageIFD.Orientation: 6}}))
    response = client.post("/v1/uploads", files={"file": ("p.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 201
    assert (response.json()["width"], response.json()["height"]) == (100, 200)


def test_the_swagger_page_sorts_its_operations_alphabetically(client: TestClient) -> None:
    """Bug: the Swagger page lists the operations in the order of the code, so a reader cannot find one."""
    page = client.get("/docs").text
    assert '"operationsSorter": "alpha"' in page and '"tagsSorter": "alpha"' in page
