"""The REST API (docs/api.md). Each route checks the request, then calls the engine."""

import asyncio
import hmac
import io
import json
import queue
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__, config, hf_models, saved_models
from .engine import Engine
from .errors import ApiError, invalid_request

PUBLIC_PATHS = frozenset({"/v1/health"})
UPLOAD_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
SSE_KEEPALIVE_SECONDS = 15


def _error_body(error: ApiError) -> dict[str, Any]:
    return {"error": error.to_dict()}


def create_app(engine: Engine, mcp_app: Any = None, mcp_lifespan: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine.start()
        try:
            if mcp_lifespan is not None:
                async with mcp_lifespan():
                    yield
            else:
                yield
        finally:
            engine.stop()

    app = FastAPI(title="mlx-imagine", version=__version__, lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(_error_body(exc), status_code=exc.status)

    @app.middleware("http")
    async def _auth(request: Request, call_next: Any) -> Any:
        token = engine.settings.token
        if token and request.url.path not in PUBLIC_PATHS:
            given = request.headers.get("authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                error = ApiError("unauthorized", 401, "The token is missing or not correct.")
                return JSONResponse(_error_body(error), status_code=401)
        return await call_next(request)

    # -- server ------------------------------------------------------------------------------------------------

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        from importlib.metadata import version

        return {"status": "ok", "server_version": __version__, "mflux_version": version("mflux")}

    @app.get("/v1/server")
    def server() -> dict[str, Any]:
        return engine.status()

    @app.get("/v1/config")
    def get_config() -> dict[str, Any]:
        return {"config": engine.settings.public_dict(), "restart_required": sorted(engine.restart_required)}

    @app.patch("/v1/config")
    def patch_config(changes: dict[str, Any] = Body(...)) -> dict[str, Any]:
        applied, restart = config.update(engine.settings, changes)
        config.save(engine.settings)
        config.make_directories(engine.settings)
        engine.restart_required.update(restart)
        if {"memory_limit_gb", "cache_limit_gb"} & set(applied):
            engine.apply_memory_limits()
        if "queue.queue_max_length" in applied:
            engine.queue.max_length = engine.settings.queue_max_length
        if "queue.job_history" in applied:
            engine.queue.set_history(engine.settings.job_history)
        return {"config": engine.settings.public_dict(), "applied": applied, "restart_required": restart}

    # -- models ------------------------------------------------------------------------------------------------

    @app.get("/v1/models/supported")
    def supported_models() -> dict[str, Any]:
        return {"models": [entry.to_dict() for entry in engine.registry.entries().values()]}

    @app.get("/v1/models/supported/{name}/schema")
    def model_schema(name: str) -> dict[str, Any]:
        from .options import schemas

        entry = engine.registry.supported(name)
        assert entry.spec is not None
        return {"name": entry.name, **schemas(entry.spec)}

    @app.get("/v1/models/huggingface")
    def huggingface_models() -> dict[str, Any]:
        return {"models": hf_models.list_models(engine.registry)}

    @app.post("/v1/models/huggingface/download", status_code=202)
    def download(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        model = body.get("model")
        if not isinstance(model, str):
            raise invalid_request("The download request needs 'model'.")
        return engine.queue.describe(engine.submit_download(model))

    @app.delete("/v1/models/huggingface/{owner}/{repo}")
    def delete_huggingface(owner: str, repo: str, confirm: bool = Query(False)) -> dict[str, Any]:
        return engine.delete_hf_model(f"{owner}/{repo}", confirm)

    @app.get("/v1/models/saved")
    def saved() -> dict[str, Any]:
        return {"models": saved_models.list_models(engine.models_dir)}

    @app.post("/v1/models/saved", status_code=202)
    def save(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_save(body))

    @app.patch("/v1/models/saved/{name}")
    def set_base(name: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        base_model = body.get("base_model")
        if not isinstance(base_model, str):
            raise invalid_request("The request needs 'base_model'.")
        return saved_models.set_base_model(engine.registry, engine.models_dir, name, base_model)

    @app.delete("/v1/models/saved/{name}")
    def delete_saved(name: str, confirm: bool = Query(False)) -> dict[str, Any]:
        return engine.delete_saved_model(name, confirm)

    # -- loaded model and generation ---------------------------------------------------------------------------

    @app.get("/v1/model")
    def loaded_model() -> Any:
        return engine.loaded.to_dict() if engine.loaded else None

    @app.post("/v1/model/load", status_code=202)
    def load(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_load(body))

    @app.post("/v1/model/unload", status_code=202)
    def unload() -> dict[str, Any]:
        return engine.queue.describe(engine.submit_unload())

    @app.post("/v1/generate", status_code=202)
    def generate(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_generate(body))

    @app.post("/v1/uploads", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = UPLOAD_TYPES.get(file.content_type or "")
        if suffix is None:
            raise invalid_request("Upload a PNG, JPEG or WebP image.", content_type=file.content_type)
        limit = engine.settings.max_upload_mb * 1000**2
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise ApiError("upload_too_large", 413, f"The upload is larger than {engine.settings.max_upload_mb} MB.")
        return await asyncio.to_thread(_store_upload, engine, data, suffix)

    @app.get("/v1/images/{job_id}/{index}")
    def image(job_id: str, index: int) -> FileResponse:
        return FileResponse(engine.image_path(job_id, index), media_type="image/png")

    # -- jobs --------------------------------------------------------------------------------------------------

    @app.get("/v1/jobs")
    def jobs(state: str | None = None, limit: int = Query(50, ge=1, le=1000)) -> dict[str, Any]:
        return {"jobs": engine.queue.list_jobs(state=state, limit=limit)}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        return engine.queue.describe(engine.queue.get(job_id))

    @app.delete("/v1/jobs/{job_id}")
    def cancel(job_id: str) -> dict[str, Any]:
        return engine.queue.describe(engine.queue.cancel(job_id))

    @app.get("/v1/jobs/{job_id}/events")
    async def events(job_id: str) -> StreamingResponse:
        job = engine.queue.get(job_id)
        return StreamingResponse(_event_stream(engine, job), media_type="text/event-stream")

    if mcp_app is not None:
        app.mount("/", mcp_app)
    return app


def _store_upload(engine: Engine, data: bytes, suffix: str) -> dict[str, Any]:
    import PIL.Image

    try:
        with PIL.Image.open(io.BytesIO(data)) as picture:
            picture.verify()
        with PIL.Image.open(io.BytesIO(data)) as picture:
            width, height = picture.size
    except Exception:  # noqa: BLE001 -- any decode failure means the file is not an image
        raise invalid_request("The upload is not a readable image.") from None
    upload_id = f"upl_{secrets.token_hex(6)}"
    input_dir = engine.input_dir
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{upload_id}{suffix}"
    path.write_bytes(data)
    return {"upload_id": upload_id, "path": str(path), "width": width, "height": height}


async def _event_stream(engine: Engine, job: Any) -> AsyncIterator[bytes]:
    subscriber = job.subscribe()
    try:
        yield _sse("state", engine.queue.describe(job))
        while True:
            try:
                event, data = await asyncio.to_thread(subscriber.get, True, SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield b": keepalive\n\n"
                continue
            if event == "end":
                yield _sse("end", data if data is not None else engine.queue.describe(job))
                return
            yield _sse(event, data)
    finally:
        job.unsubscribe(subscriber)


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
