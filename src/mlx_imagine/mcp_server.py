"""The MCP server (docs/api.md section 6). Each tool calls the same engine code as its REST route."""

import asyncio
import inspect
import json
import queue
import re
from typing import Annotated, Any, Literal

from pydantic import Field

from .engine import Engine
from .errors import ApiError, invalid_request
from .jobs import ENDED, SUCCEEDED, Job
from .options import schemas

# MCP hosts end a tool call after a limit that differs by host and version. Claude Code failed a call at about 60
# seconds, although the server sent progress notifications, so a tool never waits longer than MAX_WAIT.
DEFAULT_GENERATE_WAIT = 45
MAX_WAIT = 55
# A preview is a JPEG. A full PNG can be 2 MB or more, and an MCP client can refuse a result of that size (the Python
# client allows 1 MiB for one event). The steps are (longest side, quality), tried in order until the base64 text
# of all images of one result fits RESPONSE_BUDGET characters.
PREVIEW_STEPS = ((1024, 85), (1024, 70), (768, 70), (768, 55), (512, 55), (384, 45), (256, 40))
RESPONSE_BUDGET = 700_000
READ_ONLY = {"read_only_hint": True, "open_world_hint": False}
DESTRUCTIVE = {"destructive_hint": True, "open_world_hint": False}

_PY_TYPES: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


def jpeg_preview(source: Any, budget: int) -> bytes | None:
    """A JPEG of the image, as large as the budget (base64 characters) allows, or None when no step fits. `source`
    is a path or PNG bytes. A transparent area is drawn on white, so the preview does not show the hidden colors."""
    import io

    import PIL.Image

    with PIL.Image.open(io.BytesIO(source) if isinstance(source, bytes) else source) as opened:
        picture = opened.convert("RGBA")
    flat = PIL.Image.new("RGBA", picture.size, (255, 255, 255, 255))
    flat.alpha_composite(picture)
    rgb = flat.convert("RGB")
    for side, quality in PREVIEW_STEPS:
        candidate = rgb.copy()
        candidate.thumbnail((side, side))
        buffer = io.BytesIO()
        candidate.save(buffer, "JPEG", quality=quality)
        data = buffer.getvalue()
        if len(data) * 4 // 3 <= budget:
            return data
    return None


def _annotations(**hints: Any) -> Any:
    from mcp.types import ToolAnnotations

    return ToolAnnotations(**hints)


def _error_text(error: ApiError) -> str:
    return json.dumps({"error": error.to_dict()})


def build_mcp(engine: Engine) -> Any:
    from mcp.server import MCPServer
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.server.mcpserver.utilities.types import Image

    mcp = MCPServer(
        "mlx-imagine",
        instructions=(
            "Generate images with mflux on this machine. Call list_registry_models, then load_model, then the "
            "generate tool of the loaded model. One job runs at a time. Jobs that take long return a job ID: "
            "call get_job to read the result. To give the server an image, call upload_image and use the "
            "upload_id as image_path, mask_image or in a control. A model with the mode controlnet or inpaint "
            "needs a control image, or an image and a mask. You make the control image (canny, depth, pose): "
            "the server does not. White in a mask is drawn again: check the mask with preview_mask. Read the "
            "description of the generate tool of the model."
        ),
    )

    def call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ApiError as exc:
            raise ToolError(_error_text(exc)) from exc

    async def wait(job: Job, seconds: float, ctx: Context | None) -> dict[str, Any]:
        """Wait for the job up to `seconds`, with MCP progress notifications. Returns the job object."""
        seconds = min(seconds, MAX_WAIT)
        if seconds <= 0:
            return engine.queue.describe(job)
        subscriber = job.subscribe()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        try:
            while job.state not in ENDED and loop.time() < deadline:
                try:
                    event, data = await asyncio.to_thread(subscriber.get, True, min(1.0, deadline - loop.time()))
                except queue.Empty:
                    continue
                if event == "progress" and ctx is not None and data:
                    try:
                        await _report(ctx, data)
                    except ValueError:  # no request context: a direct call, not a client request
                        ctx = None
                if event == "end":
                    break
        finally:
            job.unsubscribe(subscriber)
        return engine.queue.describe(job)

    def job_content(described: dict[str, Any], with_images: bool) -> list[Any]:
        content: list[Any] = [json.dumps(described)]
        if described["state"] in ("queued", "running"):
            content.append(
                f"The job is not done. Call get_job with job_id {described['id']!r} to read the result. "
                f"Set wait_seconds (up to {MAX_WAIT}) to wait for it."
            )
        if with_images and described["state"] == SUCCEEDED and described["type"] == "generate":
            found = (described["result"] or {}).get("images", [])
            content += [
                Image(data=jpeg_preview(item["path"], RESPONSE_BUDGET // len(found)), format="jpeg") for item in found
            ]
        return content

    # -- server and models -------------------------------------------------------------------------------------

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def server_status() -> dict[str, Any]:
        """The loaded model, the target model, the queue length and the memory use."""
        return engine.status()

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def get_config() -> dict[str, Any]:
        """All server settings. The API token shows as ***."""
        return {"config": engine.settings.public_dict(), "restart_required": sorted(engine.restart_required)}

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    def set_config(changes: dict[str, Any]) -> dict[str, Any]:
        """Change server settings. Directory, host, port and token changes apply after a restart."""
        applied, restart = call(engine.update_settings, changes)
        return {"applied": applied, "restart_required": restart}

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def list_registry_models() -> list[dict[str, Any]]:
        """All mflux models, with `supported` true for the models this server can run."""
        return [entry.to_dict() for entry in engine.registry.entries().values()]

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def list_models(
        view: Literal["models", "mflux", "hf", "hf/all", "other", "all"] = "models",
        supported: bool | None = None,
        family: str | None = None,
        q: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List the models in the Hugging Face cache. `view` models (the default) is the image models in both
        formats. mflux is the mflux-format models. hf is the original layout of the mflux registry models, 1 GB
        or larger. hf/all also has the hf repositories under 1 GB (small files only). other is all other
        repositories. all is mflux, hf and other."""
        filters = {"supported": supported, "family": family, "q": q, "limit": limit, "cursor": cursor}
        return call(engine.catalog.list_models, view, filters)

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def list_community_models(
        quant: str | None = None,
        ns: str | None = None,
        search: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List the mflux-community models on the Hub, from the catalog the server syncs. `quant` is 3, 4, 5, 6,
        8 or bf16. `ns` is the upstream owner, for example Qwen, krea or bfl. `search` is a text in the
        repository ID, for example qwen. Download one with download_model."""
        filters = {"quant": quant, "ns": ns, "search": search, "limit": limit, "cursor": cursor}
        return call(engine.catalog.list_community, filters)

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=True))
    async def download_model(
        model: str, ctx: Context, wait_seconds: float = 0, confirm: bool = False
    ) -> dict[str, Any]:
        """Download an mflux model from Hugging Face. Returns the job. Read the result with get_job.
        If the server answers confirm_required, ask the user, then send confirm as true."""
        return await wait(call(engine.submit_download, model, confirm), wait_seconds, ctx)

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    async def save_model(model: str, quantize: Literal[3, 4, 5, 6, 8], ctx: Context, name: str | None = None,
                         wait_seconds: float = 0) -> dict[str, Any]:  # fmt: skip
        """Save a quantized copy of a model to the Hugging Face cache as local/<name>. The save unloads the
        loaded model."""
        request = {"model": model, "quantize": quantize, **({"name": name} if name else {})}
        return await wait(call(engine.submit_save, request), wait_seconds, ctx)

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    def set_saved_model_base(repo_id: str, base_model: str) -> dict[str, Any]:
        """Record which mflux model an mflux-format model is. Use it for a model with no base model."""
        return call(engine.set_base_model, repo_id, base_model)

    @mcp.tool(annotations=_annotations(**DESTRUCTIVE))
    def delete_model(repo_id: str, confirm: bool) -> dict[str, Any]:
        """Delete an image model from the Hugging Face cache. Set confirm to true."""
        return call(engine.delete_model, repo_id, confirm)

    # -- images in -------------------------------------------------------------------------------------------

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    def upload_image(data: str, subdir: str | None = None) -> dict[str, Any]:
        """Give the server an image: PNG, JPEG or WebP as base64 text, without a data: prefix. Returns the
        upload_id. Use it as image_path, mask_image or controlnet_image_path, or in a control. `subdir` is a
        folder of the input directory."""
        import base64
        import binascii

        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise ToolError(
                _error_text(invalid_request("'data' must be base64 text, without a data: prefix."))
            ) from None
        return call(engine.store_upload, raw, None, subdir)

    # -- mask preview -----------------------------------------------------------------------------------------

    @mcp.tool(annotations=_annotations(**READ_ONLY), structured_output=False)
    def preview_mask(image_path: str, mask_image: str) -> list[Any]:
        """Draw the mask in red on the image and return the PNG, so you can check the mask before you generate.
        White in the mask is the area that the model draws again. Both arguments are upload IDs or paths inside
        the input directory. It needs no model."""
        png = call(engine.preview_mask, image_path, mask_image)
        data = jpeg_preview(png, RESPONSE_BUDGET)
        return [Image(data=data, format="jpeg")] if data else ["The preview is too large to send."]

    # -- LoRA files -------------------------------------------------------------------------------------------

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def list_loras() -> dict[str, Any]:
        """The LoRA files in the LoRA directory. Use a name in the `loras` argument of load_model."""
        return {"directory": str(engine.lora_dir), "loras": engine.list_loras()}

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=True))
    async def download_lora(
        repo_id: str, filename: str, ctx: Context, wait_seconds: float = 0, confirm: bool = False
    ) -> dict[str, Any]:
        """Download one .safetensors LoRA file from a Hugging Face repository. Returns the job. If the server
        answers confirm_required, ask the user, then send confirm as true."""
        return await wait(call(engine.submit_lora_download, repo_id, filename, confirm), wait_seconds, ctx)

    @mcp.tool(annotations=_annotations(**DESTRUCTIVE))
    def delete_lora(name: str, confirm: bool) -> dict[str, Any]:
        """Delete a LoRA file from the LoRA directory. Set confirm to true."""
        return call(engine.delete_lora, name, confirm)

    # -- loaded model and jobs ---------------------------------------------------------------------------------

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    async def load_model(model: str, ctx: Context, quantize: Literal[3, 4, 5, 6, 8] | None = None,
                         low_ram: bool = False, base_model: str | None = None,
                         loras: list[dict[str, Any]] | None = None,
                         wait_seconds: float = 0) -> dict[str, Any]:  # fmt: skip
        """Load a model: a registry name, an alias, or the name of a saved model. Unloads the old model.
        `loras` is a list of {name, scale} from list_loras. It needs a model that supports LoRA."""
        request: dict[str, Any] = {"model": model, "low_ram": low_ram}
        if quantize is not None:
            request["quantize"] = quantize
        if base_model is not None:
            request["base_model"] = base_model
        if loras is not None:
            request["loras"] = loras
        return await wait(call(engine.submit_load, request), wait_seconds, ctx)

    @mcp.tool(annotations=_annotations(destructive_hint=False, open_world_hint=False))
    def unload_model() -> dict[str, Any]:
        """Unload the model and free its memory."""
        return engine.queue.describe(call(engine.submit_unload))

    @mcp.tool(annotations=_annotations(**READ_ONLY))
    def list_jobs(state: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """The jobs that wait, the job that runs, and the last jobs that ended."""
        return engine.queue.list_jobs(state=state, limit=limit)

    @mcp.tool(annotations=_annotations(**READ_ONLY), structured_output=False)
    async def get_job(job_id: str, ctx: Context, wait_seconds: float = 0) -> list[Any]:
        """Get a job. A generate job that succeeded also returns a preview of each image (JPEG, 1024 pixels at most).
        The full image is the path in the job result."""
        job = call(engine.queue.get, job_id)
        return job_content(await wait(job, wait_seconds, ctx), with_images=True)

    @mcp.tool(annotations=_annotations(**DESTRUCTIVE))
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a job. A job that waits leaves the queue. A job that runs stops at the next step."""
        return engine.queue.describe(call(engine.queue.cancel, job_id))

    # -- one generate tool for each supported model ------------------------------------------------------------

    for entry in engine.registry.entries().values():
        if entry.supported:
            _add_generate_tool(mcp, engine, entry, call, wait, job_content, Context)
    return mcp


async def _report(ctx: Any, data: dict[str, Any]) -> None:
    if "step" in data:
        message = f"image {data['image_index'] + 1}/{data['image_count']}, step {data['step']}/{data['total_steps']}"
        await ctx.report_progress(data["step"], data["total_steps"], message=message)
    elif "bytes_done" in data and data.get("bytes_total"):
        await ctx.report_progress(data["bytes_done"], data["bytes_total"], message=data.get("file"))
    elif "phase" in data:
        await ctx.report_progress(0, None, message=data["phase"])


def tool_name(model_name: str) -> str:
    return "generate_" + re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _python_type(schema: dict[str, Any]) -> Any:
    if "enum" in schema:
        return Literal.__getitem__(tuple(schema["enum"]))
    if schema.get("type") == "array":
        return list.__class_getitem__(_python_type(schema["items"]))
    if "anyOf" in schema:
        return int | str
    return _PY_TYPES.get(schema.get("type", "string"), str)


def generate_signature(generate_schema: dict[str, Any], context_type: Any) -> inspect.Signature:
    required = set(generate_schema.get("required", []))
    parameters = []
    for name, schema in generate_schema["properties"].items():
        annotation = _python_type(schema)
        description = schema.get("description")
        if name not in required:
            annotation = annotation | None
        if description:
            annotation = Annotated[annotation, Field(description=description)]
        default = inspect.Parameter.empty if name in required else None
        parameters.append(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
        )
    wait_annotation = Annotated[float, Field(description="Seconds to wait for the images. 0 returns the job at once.")]
    parameters.append(
        inspect.Parameter(
            "wait_seconds", inspect.Parameter.KEYWORD_ONLY, default=DEFAULT_GENERATE_WAIT, annotation=wait_annotation
        )
    )
    parameters.append(inspect.Parameter("ctx", inspect.Parameter.KEYWORD_ONLY, annotation=context_type))
    # Required parameters first: Python signatures do not allow a required one after a default.
    parameters.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return inspect.Signature(parameters, return_annotation=list[Any])


def _add_generate_tool(
    mcp: Any, engine: Engine, entry: Any, call: Any, wait: Any, job_content: Any, context_type: Any
) -> None:
    generate_schema = schemas(entry.spec)["generate_schema"]
    model_name = entry.name
    notes = getattr(entry.spec.build_parser(), "server_notes", "")

    async def generate(**kwargs: Any) -> list[Any]:
        ctx = kwargs.pop("ctx", None)
        wait_seconds = kwargs.pop("wait_seconds", DEFAULT_GENERATE_WAIT)
        request = {name: value for name, value in kwargs.items() if value is not None}
        target = engine.target
        if target is None or target.entry.name != model_name:
            call(engine.submit_load, {"model": model_name})
        job = call(engine.submit_generate, request)
        return job_content(await wait(job, wait_seconds, ctx), with_images=True)

    signature = generate_signature(generate_schema, context_type)
    setattr(generate, "__signature__", signature)
    # MCP finds the Context parameter through typing.get_type_hints(), which reads __annotations__.
    generate.__annotations__ = {p.name: p.annotation for p in signature.parameters.values()}
    generate.__name__ = tool_name(model_name)
    mcp.add_tool(
        generate,
        name=tool_name(model_name),
        description=(
            f"Generate images with {model_name}. Loads the model first if it is not the loaded model. "
            "Waits up to wait_seconds, then returns the job and the images, or the job ID to read with get_job."
            + (f" {notes}" if notes else "")
        ),
        annotations=_annotations(destructive_hint=False, open_world_hint=False),
        structured_output=False,
    )
