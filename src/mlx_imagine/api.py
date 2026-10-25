"""The REST API (docs/api.md). Each route checks the request, then calls the engine."""

import asyncio
import json
import queue
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Body, Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer

from . import __version__
from .engine import Engine
from .errors import ApiError, invalid_request
from .middleware import AuthMiddleware, BodyLimitMiddleware

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
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

    swagger = engine.settings.swagger_enable
    api_enabled = engine.settings.api_enable  # read once: a change to services.* waits for a restart
    app = FastAPI(
        title="mlx-imagine",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if swagger else None,
        redoc_url="/redoc" if swagger else None,
        openapi_url="/openapi.json" if swagger else None,
        # The Swagger page lists the operations by path, then method, and the tag groups by name.
        swagger_ui_parameters={"operationsSorter": "alpha", "tagsSorter": "alpha"},
        # Adds the bearer scheme to the schema, so the Swagger page shows an Authorize button. The middleware
        # below checks the token; this dependency only describes it.
        dependencies=[Depends(HTTPBearer(auto_error=False))] if swagger else [],
    )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(_error_body(exc), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"] if part not in ("query", "path", "body"))
        error = ApiError("invalid_request", 400, f"{where}: {first['msg']}" if where else first["msg"])
        return JSONResponse(_error_body(error), status_code=400)

    # Added last, so it runs first: the body limit sees a request before the auth check reads its headers.
    app.add_middleware(
        AuthMiddleware, get_token=lambda: engine.settings.token, api_enabled=api_enabled, swagger=swagger
    )
    app.add_middleware(BodyLimitMiddleware, get_upload_mb=lambda: engine.settings.max_upload_mb)

    from . import openai_api

    openai_api.register(app, engine)

    # -- server ------------------------------------------------------------------------------------------------

    @app.get("/v8/health")
    def health() -> dict[str, Any]:
        from importlib.metadata import version

        return {"status": "ok", "server_version": __version__, "mflux_version": version("mflux")}

    @app.get("/v8/server")
    def server() -> dict[str, Any]:
        return engine.status()

    @app.get("/v8/config")
    def get_config() -> dict[str, Any]:
        return {"config": engine.settings.public_dict(), "restart_required": sorted(engine.restart_required)}

    @app.patch("/v8/config")
    def patch_config(
        changes: dict[str, Any] = Body(..., examples=[{"queue": {"queue_max_length": 50}, "cache_limit_gb": 4}]),
    ) -> dict[str, Any]:
        applied, restart = engine.update_settings(changes)
        return {"config": engine.settings.public_dict(), "applied": applied, "restart_required": restart}

    # -- models ------------------------------------------------------------------------------------------------

    def model_list(view: str, description: str) -> None:
        @app.get(f"/v8/models{'' if view == 'models' else '/' + view}", summary=description)
        def list_models(
            supported: Annotated[bool | None, Query(description="true: only models this server can run.")] = None,
            family: Annotated[str | None, Query(description="A model family, for example flux2 or z-image.")] = None,
            q: Annotated[str | None, Query(description="A text in the repository ID.")] = None,
            limit: Annotated[int | None, Query(ge=1, le=100, description="The page size.")] = None,
            cursor: Annotated[str | None, Query(description="The next_cursor of the last page.")] = None,
        ) -> dict[str, Any]:
            filters = {"supported": supported, "family": family, "q": q, "limit": limit, "cursor": cursor}
            return engine.catalog.list_models(view, filters)

    model_list("models", "List the image models: mflux format and Hugging Face format")
    model_list("mflux", "List the mflux-format models")
    model_list("hf", "List the Hugging Face-format models of the mflux registry, 1 GB or larger")
    model_list("hf/all", "List the Hugging Face-format models of the mflux registry, all sizes")
    model_list("other", "List the other repositories in the Hugging Face cache")
    model_list("all", "List all repositories in the Hugging Face cache, except hf repositories under 1 GB")

    @app.get("/v8/models-community", summary="List the mflux-community models on the Hub")
    def community_models(
        q: Annotated[str | None, Query(description="The quantization: 3, 4, 5, 6, 8 or bf16. Same as quant.")] = None,
        quant: Annotated[str | None, Query(description="The quantization: 3, 4, 5, 6, 8, q4 or bf16.")] = None,
        ns: Annotated[str | None, Query(description="The upstream owner, for example Qwen, krea or bfl.")] = None,
        namespace: Annotated[str | None, Query(description="Same as ns.")] = None,
        search: Annotated[str | None, Query(description="A text in the repository ID, for example qwen.")] = None,
        s: Annotated[str | None, Query(description="Same as search.")] = None,
        limit: Annotated[int | None, Query(ge=1, le=100, description="The page size.")] = None,
        cursor: Annotated[str | None, Query(description="The next_cursor of the last page.")] = None,
    ) -> dict[str, Any]:
        filters = {"quant": quant or q, "ns": ns or namespace, "search": search or s, "limit": limit, "cursor": cursor}
        return engine.catalog.list_community(filters)

    @app.post("/v8/models/scans", status_code=202)
    def rescan() -> dict[str, Any]:
        engine.catalog.scan_all()
        return engine.catalog.status()

    @app.get("/v8/registry")
    def registry_models() -> dict[str, Any]:
        return {"models": [entry.to_dict() for entry in engine.registry.entries().values()]}

    @app.get("/v8/registry/{name}/schema")
    def model_schema(name: str) -> dict[str, Any]:
        from .options import schemas

        entry = engine.registry.supported(name)
        assert entry.spec is not None
        return {"name": entry.name, **schemas(entry.spec)}

    @app.post("/v8/downloads", status_code=202)
    def download(
        body: dict[str, Any] = Body(
            ...,
            examples=[
                {"kind": "model", "model": "z-image-turbo", "confirm": True},
                {"kind": "lora", "repo_id": "owner/name", "filename": "style.safetensors", "confirm": True},
            ],
        ),
    ) -> dict[str, Any]:
        kind = body.get("kind", "model")
        if kind == "lora":
            job = engine.submit_lora_download(body.get("repo_id"), body.get("filename"), body.get("confirm") is True)
            return engine.queue.describe(job)
        if kind != "model":
            raise invalid_request("The download kind must be 'model' or 'lora'.", kind=kind)
        model = body.get("model")
        if not isinstance(model, str):
            raise invalid_request("The download request needs 'model'.")
        return engine.queue.describe(engine.submit_download(model, body.get("confirm") is True))

    @app.post("/v8/saves", status_code=202)
    def save(
        body: dict[str, Any] = Body(
            ..., examples=[{"model": "z-image-turbo", "quantize": 8, "name": "z-image-turbo-q8"}]
        ),
    ) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_save(body))

    @app.get("/v8/models/{owner}/{name}")
    def get_model(owner: str, name: str) -> dict[str, Any]:
        return engine.catalog.get(f"{owner}/{name}")

    @app.patch("/v8/models/{owner}/{name}")
    def set_base(
        owner: str, name: str, body: dict[str, Any] = Body(..., examples=[{"base_model": "z-image-turbo"}])
    ) -> dict[str, Any]:
        base_model = body.get("base_model")
        if not isinstance(base_model, str):
            raise invalid_request("The request needs 'base_model'.")
        return engine.set_base_model(f"{owner}/{name}", base_model)

    @app.delete("/v8/models/{owner}/{name}")
    def delete_model(
        owner: str, name: str, confirm: bool = Query(False, description="Must be true.")
    ) -> dict[str, Any]:
        return engine.delete_model(f"{owner}/{name}", confirm)

    # -- LoRA files --------------------------------------------------------------------------------------------

    @app.get("/v8/loras")
    def list_loras() -> dict[str, Any]:
        return {"directory": str(engine.lora_dir), "loras": engine.list_loras()}

    @app.delete("/v8/loras/{name:path}")
    def delete_lora(name: str, confirm: bool = Query(False, description="Must be true.")) -> dict[str, Any]:
        return engine.delete_lora(name, confirm)

    # -- loaded model and generation ---------------------------------------------------------------------------

    @app.get("/v8/engine/model")
    def loaded_model() -> dict[str, Any]:
        if engine.loaded is None:
            raise ApiError("model_not_loaded", 404, "No model is loaded. Send PUT /v8/engine/model first.")
        return engine.loaded.to_dict()

    @app.put("/v8/engine/model", status_code=202)
    def load(
        body: dict[str, Any] = Body(
            ...,
            examples=[
                {
                    "model": "z-image-turbo",
                    "quantize": 8,
                    "low_ram": False,
                    "loras": [{"name": "style.safetensors", "scale": 0.8}],
                }
            ],
        ),
    ) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_load(body))

    @app.delete("/v8/engine/model", status_code=202)
    def unload() -> dict[str, Any]:
        return engine.queue.describe(engine.submit_unload())

    @app.post("/v8/generate", status_code=202)
    def generate(
        body: dict[str, Any] = Body(
            ...,
            examples=[{"prompt": "a cat on a red chair", "seed": [1, 2], "steps": 8, "width": 1024, "height": 1024}],
        ),
    ) -> dict[str, Any]:
        return engine.queue.describe(engine.submit_generate(body))

    @app.post("/v8/uploads", status_code=201)
    async def upload(
        file: UploadFile = File(...),
        subdir: Annotated[str | None, Query(description="A folder inside the input directory.")] = None,
    ) -> dict[str, Any]:
        suffix = UPLOAD_TYPES.get(file.content_type or "")
        if suffix is None:
            raise invalid_request("Upload a PNG, JPEG or WebP image.", content_type=file.content_type)
        limit = engine.settings.max_upload_mb * 1000**2
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise ApiError("upload_too_large", 413, f"The upload is larger than {engine.settings.max_upload_mb} MB.")
        return await asyncio.to_thread(engine.store_upload, data, suffix, subdir)

    @app.post("/v8/masks/preview")
    async def preview_mask(
        body: dict[str, Any] = Body(..., examples=[{"image_path": "upl_01J9Z4", "mask_image": "upl_01J9Z5"}]),
    ) -> Response:
        image_path, mask_image = body.get("image_path"), body.get("mask_image")
        if not isinstance(image_path, str) or not isinstance(mask_image, str):
            raise invalid_request("The request needs 'image_path' and 'mask_image'.")
        png = await asyncio.to_thread(engine.preview_mask, image_path, mask_image)
        return Response(png, media_type="image/png")

    @app.get("/v8/images/{job_id}/{index}")
    def image(job_id: str, index: int) -> FileResponse:
        return FileResponse(engine.image_path(job_id, index), media_type="image/png")

    # -- jobs --------------------------------------------------------------------------------------------------

    @app.get("/v8/jobs")
    def jobs(
        state: JobState = None,  # ty: ignore[invalid-parameter-default]
        limit: int = Query(50, ge=1, le=1000),
    ) -> dict[str, Any]:
        return {"jobs": engine.queue.list_jobs(state=state, limit=limit)}

    @app.get("/v8/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        return engine.queue.describe(engine.queue.get(job_id))

    @app.delete("/v8/jobs/{job_id}")
    def cancel(job_id: str) -> dict[str, Any]:
        return engine.queue.describe(engine.queue.cancel(job_id))

    @app.get("/v8/jobs/{job_id}/events")
    async def events(job_id: str) -> StreamingResponse:
        job = engine.queue.get(job_id)
        return StreamingResponse(_event_stream(engine, job), media_type="text/event-stream")

    if mcp_app is not None:
        app.mount("/", mcp_app)
    return app


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
