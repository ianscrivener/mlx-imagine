"""A thin OpenAI-compatible surface over the job queue: GET /v1/models, POST /v1/images/generations and
POST /v1/images/edits. Many clients speak this API. Each route builds a load job and a generate job with the same
engine methods as the native routes, and waits for the images, as the OpenAI API does."""

import asyncio
import base64
import re
import time
from typing import Any

from fastapi import Body, FastAPI, File, Form, Request, UploadFile

from .engine import Engine
from .errors import ApiError, invalid_request
from .jobs import ENDED, SUCCEEDED

SIZE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
POLL_SECONDS = 0.25


def parse_size(size: str | None) -> dict[str, int]:
    if size in (None, "", "auto"):
        return {}
    match = SIZE.match(str(size))
    if not match:
        raise invalid_request("size must look like 1024x1024.", size=size)
    return {"width": int(match[1]), "height": int(match[2])}


def build_request(body: dict[str, Any]) -> dict[str, Any]:
    """The native generate request for an OpenAI images request."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise invalid_request("The request needs 'prompt'.")
    count = body.get("n", 1)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise invalid_request("n must be a whole number of 1 or more.", n=count)
    request: dict[str, Any] = {"prompt": prompt, **parse_size(body.get("size"))}
    for name in (
        "seed",
        "steps",
        "guidance",
        "negative_prompt",
        "input_subdir",
        "output_subdir",
    ):  # extensions: the OpenAI API has no such field
        if body.get(name) is not None:
            request[name] = body[name]
    if "seed" in request and not isinstance(request["seed"], list):
        request["seed"] = [request["seed"]]
    if "seed" not in request and count > 1:
        import random

        request["seed"] = [random.randrange(2**32) for _ in range(count)]
    elif "seed" in request and len(request["seed"]) != count and count > 1:
        raise invalid_request("The number of seeds must equal n.", n=count)
    return request


def _ensure_model(engine: Engine, model: str | None) -> None:
    target = engine.target
    if model is None:
        if target is None:
            raise invalid_request("No model is loaded. Give 'model'.")
        return
    if target is not None and model in (target.to_dict()["name"], target.entry.name):
        return
    engine.submit_load({"model": model})


async def _run(request: Request, engine: Engine, model: str | None, native: dict[str, Any]) -> dict[str, Any]:
    _ensure_model(engine, model)
    job = engine.submit_generate(native)
    try:
        while job.state not in ENDED:
            if await request.is_disconnected():
                engine.queue.cancel(job.id)
                raise ApiError("client_disconnected", 499, "The client closed the request.")
            await asyncio.sleep(POLL_SECONDS)
    except asyncio.CancelledError:
        engine.queue.cancel(job.id)
        raise
    described = engine.queue.describe(job)
    if described["state"] != SUCCEEDED:
        error = described.get("error") or {}
        raise ApiError(error.get("code", "generation_failed"), 500, error.get("message", "The job failed."))
    return described


def _images(request: Request, described: dict[str, Any], response_format: str) -> list[dict[str, str]]:
    items = []
    for image in described["result"]["images"]:
        if response_format == "url":
            items.append({"url": f"{str(request.base_url).rstrip('/')}{image['url']}"})
        else:
            with open(image["path"], "rb") as file:
                items.append({"b64_json": base64.b64encode(file.read()).decode()})
    return items


def _response_format(value: Any) -> str:
    if value not in (None, "b64_json", "url"):
        raise invalid_request("response_format must be 'b64_json' or 'url'.", response_format=value)
    return value or "b64_json"


def register(app: FastAPI, engine: Engine) -> None:
    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        models = engine.catalog.list_models({"location": "local", "supported": True})["models"]
        names = sorted({m["name"] if m["store"] == "mflux_dir" else (m["model_name"] or m["name"]) for m in models})
        created = int(time.time())
        data = [{"id": n, "object": "model", "created": created, "owned_by": "mlx-imagine"} for n in names]
        return {"object": "list", "data": data}

    @app.post("/v1/images/generations")
    async def generations(
        request: Request,
        body: dict[str, Any] = Body(
            ...,
            examples=[{"model": "z-image-turbo", "prompt": "a cat on a red chair", "n": 1, "size": "1024x1024"}],
        ),
    ) -> dict[str, Any]:
        fmt = _response_format(body.get("response_format"))
        described = await _run(request, engine, body.get("model"), build_request(body))
        return {"created": int(time.time()), "data": _images(request, described, fmt)}

    @app.post("/v1/images/edits")
    async def edits(
        request: Request,
        image: UploadFile = File(...),
        prompt: str = Form(...),
        model: str | None = Form(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str | None = Form(None),
        output_subdir: str | None = Form(None),
    ) -> dict[str, Any]:
        from .api import UPLOAD_TYPES

        fmt = _response_format(response_format)
        suffix = UPLOAD_TYPES.get(image.content_type or "")
        if suffix is None:
            raise invalid_request("Upload a PNG, JPEG or WebP image.", content_type=image.content_type)
        limit = engine.settings.max_upload_mb * 1000**2
        data = await image.read(limit + 1)
        if len(data) > limit:
            raise ApiError("upload_too_large", 413, f"The upload is larger than {engine.settings.max_upload_mb} MB.")
        stored = await asyncio.to_thread(engine.store_upload, data, suffix)
        native = build_request({"prompt": prompt, "n": n, "size": size, "output_subdir": output_subdir})
        _ensure_model(engine, model)
        modes = engine.target.entry.modes if engine.target else ()
        native["image_paths" if "edit" in modes else "image_path"] = (
            [stored["path"]] if "edit" in modes else stored["path"]
        )
        described = await _run(request, engine, None, native)
        return {"created": int(time.time()), "data": _images(request, described, fmt)}
