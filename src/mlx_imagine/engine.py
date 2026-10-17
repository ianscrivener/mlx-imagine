"""The server state and the job handlers: the loaded model, load, unload, generate, download and save.

Every handler here runs on the job queue's worker thread. The submit_* methods run on the client threads: they
check a request against the target model (docs/api.md 2.3) and add a job, but do no MLX work."""

import gc
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import hf_models, saved_models
from .catalog import Catalog
from .config import Settings
from .errors import ApiError, conflict, invalid_request, not_found, unprocessable
from .jobs import Job, JobCancelled, JobQueue
from .options import SERVER_OPTIONS, check_image_paths, generate_names, load_names, parse, to_argv
from .paths import resolve_inside, safe_name
from .registry import CommandSpec, ModelEntry, Registry

logger = logging.getLogger(__name__)

GB = 1000**3
# A prompt cache entry holds the text features of one prompt. Clear the cache when it grows past this size.
PROMPT_CACHE_LIMIT = 8
# Keep this much of the system memory free when the server checks if a model fits.
MEMORY_HEADROOM = 0.10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class LoadSpec:
    """What a load request asked for, checked and turned into the command-line options of its command."""

    entry: ModelEntry
    request: dict[str, Any]
    argv: list[str]
    quantize: int | None
    low_ram: bool
    saved_name: str | None = None
    # An mflux-format model in the Hugging Face cache: its repository ID and its snapshot folder.
    hub_repo: str | None = None
    model_path: Path | None = None

    @property
    def spec(self) -> CommandSpec:
        assert self.entry.spec is not None
        return self.entry.spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.saved_name or self.hub_repo or self.entry.name,
            "model_name": self.entry.name,
            "saved": self.saved_name is not None,
            "quantize": self.quantize,
            "low_ram": self.low_ram,
        }


@dataclass
class Loaded:
    spec: LoadSpec
    model: Any
    base_tiling: Any
    state: str = "ready"
    loaded_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {**self.spec.to_dict(), "state": self.state, "loaded_at": self.loaded_at}


class ServerCallback:
    """Registered once on each loaded model (the callback registry has no unregister call). It reports the
    denoising step of the job that runs, and stops the run when a client cancels the job."""

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine

    def _stop_if_cancelled(self) -> None:
        job = self._engine.current_job
        if job is not None and job.cancelled:
            from mflux.utils.exceptions import StopImageGenerationException

            raise StopImageGenerationException("A client cancelled the job.")

    def call_before_loop(self, **kwargs: Any) -> None:
        self._stop_if_cancelled()

    def call_in_loop(self, t: int, seed: int, prompt: str, latents: Any, config: Any, time_steps: Any) -> None:
        job = self._engine.current_job
        if job is not None:
            index, count = self._engine.image_position
            total = getattr(config, "num_inference_steps", None) or len(time_steps)
            job.set_progress({"image_index": index, "image_count": count, "step": t + 1, "total_steps": total})
        self._stop_if_cancelled()


class Engine:
    def __init__(self, settings: Settings, registry: Registry | None = None, queue: JobQueue | None = None) -> None:
        self.settings = settings
        self.registry = registry or Registry()
        self.queue = queue or JobQueue(settings.queue_max_length, settings.job_history)
        self.catalog = Catalog(settings, self.registry)
        self.loaded: Loaded | None = None
        self.target: LoadSpec | None = None
        self.current_job: Job | None = None
        self.image_position: tuple[int, int] = (0, 0)
        self.restart_required: set[str] = set()
        self._default_cache_limit: int | None = None

    # -- start and stop ----------------------------------------------------------------------------------------

    def start(self) -> None:
        import mlx.core as mx

        # set_cache_limit returns the limit it replaced: read MLX's default, then put it back.
        self._default_cache_limit = mx.set_cache_limit(0)
        mx.set_cache_limit(self._default_cache_limit)
        self.apply_memory_limits()
        self.queue.start()
        self.catalog.start()

    def stop(self) -> None:
        self.catalog.stop()
        self.queue.stop(timeout=5)

    def update_settings(self, changes: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Check, save and apply a settings change. The REST API and the MCP tool both call this, so the live
        queue and memory limits change the same way from both. Returns (applied now, applied after restart)."""
        from . import config

        applied, restart = config.update(self.settings, changes)
        config.save_changes(changes)
        config.make_directories(self.settings)
        self.restart_required.update(restart)
        if {"memory_limit_gb", "cache_limit_gb"} & set(applied):
            self.apply_memory_limits()
        if "queue.queue_max_length" in applied:
            self.queue.max_length = self.settings.queue_max_length
        if {"mflux_models.cron_minutes", "huggingface_models.cron_minutes"} & set(applied):
            self.catalog.start()  # a start with both intervals at 0 made no thread; the thread reads the new ones
        if "queue.job_history" in applied:
            self.queue.set_history(self.settings.job_history)
        return applied, restart

    def apply_memory_limits(self) -> None:
        import mlx.core as mx

        # The server owns this process, so the limits are its to set (docs/scope.md 3.8).
        if self.settings.memory_limit_gb is not None:
            mx.set_memory_limit(int(self.settings.memory_limit_gb * GB))
        if self.settings.cache_limit_gb is not None:
            mx.set_cache_limit(int(self.settings.cache_limit_gb * GB))
        elif self._default_cache_limit is not None:
            mx.set_cache_limit(self._default_cache_limit)

    # -- state -------------------------------------------------------------------------------------------------

    def memory(self) -> dict[str, Any]:
        import mlx.core as mx

        def gb(value: float | None) -> float | None:
            return None if value is None else round(value / GB, 2)

        return {
            "system_total_gb": gb(system_memory()),
            "mlx_active_gb": gb(mx.get_active_memory()),
            "mlx_peak_gb": gb(mx.get_peak_memory()),
            "mlx_cache_gb": gb(mx.get_cache_memory()),
            "mlx_memory_limit_gb": self.settings.memory_limit_gb,
            "mlx_cache_limit_gb": self.settings.cache_limit_gb,
        }

    def status(self) -> dict[str, Any]:
        import platform

        running = self.queue.running
        return {
            "platform": f"{platform.system().lower()}-{platform.machine()}",
            "loaded_model": self.loaded.to_dict() if self.loaded else None,
            "target_model": self.target.to_dict() if self.target else None,
            "running_job": running.id if running else None,
            "queue_length": len(self.queue.waiting()),
            "memory": self.memory(),
            "restart_required": bool(self.restart_required),
        }

    def _hub_repos_in_use(self) -> set[str]:
        specs = [s for s in (self.target, self.loaded.spec if self.loaded else None) if s]
        jobs = self.queue.waiting() + ([self.queue.running] if self.queue.running else [])
        specs += [job.context["load_spec"] for job in jobs if isinstance(job.context.get("load_spec"), LoadSpec)]
        return {spec.hub_repo.lower() for spec in specs if spec.hub_repo}

    def _models_in_use(self) -> tuple[set[str], set[str]]:
        """(registry names, saved model names) that are loaded, are the target, or that a waiting job uses."""
        names, saved = set(), set()
        specs = [s for s in (self.target, self.loaded.spec if self.loaded else None) if s]
        for job in self.queue.waiting() + ([self.queue.running] if self.queue.running else []):
            if isinstance(job.context.get("load_spec"), LoadSpec):
                specs.append(job.context["load_spec"])
            if job.type in {"download", "save"} and job.request.get("model"):
                names.add(str(job.request["model"]))
        for spec in specs:
            names.add(spec.entry.name)
            if spec.saved_name:
                saved.add(spec.saved_name)
        return names, saved

    # -- load ----------------------------------------------------------------------------------------------------

    def build_load_spec(self, request: dict[str, Any]) -> LoadSpec:
        request = dict(request)
        model = request.pop("model", None)
        if not isinstance(model, str) or not model:
            raise invalid_request("The load request needs 'model'.")
        base_model = request.pop("base_model", None)
        saved_name = self._saved_name(model)
        hub = self._hub_mflux_model(model)
        if hub is not None:
            if "quantize" in request:
                raise invalid_request("An mflux-format model is already quantized. Do not give 'quantize'.")
            snapshot, source = hub
            entry = self.registry.supported(base_model or self._hub_base_model(model, source))
            argv = ["--model", str(snapshot), "--base-model", entry.name]
        elif saved_name is not None:
            if "quantize" in request:
                raise invalid_request("A saved model is already quantized. Do not give 'quantize'.")
            entry = self.registry.supported(saved_models.base_model_of(self.models_dir, saved_name, base_model))
            argv = ["--model", str(self.models_dir / saved_name), "--base-model", entry.name]
        else:
            entry = self.registry.supported(model)
            argv = ["--model", entry.name]
            if base_model is not None:
                argv += ["--base-model", str(base_model)]
            if "quantize" not in request and self.settings.default_quantize is not None:
                request["quantize"] = self.settings.default_quantize
            self._check_quantize(request.get("quantize"))
        spec = entry.spec
        assert spec is not None
        argv += to_argv(spec.build_parser(), request, allowed=load_names(spec))
        load_spec = LoadSpec(
            entry=entry,
            request={"model": model, **({"base_model": base_model} if base_model else {}), **request},
            argv=argv,
            quantize=request.get("quantize"),
            low_ram=bool(request.get("low_ram", False)),
            saved_name=saved_name,
            hub_repo=model if hub is not None else None,
            model_path=hub[0] if hub is not None else None,
        )
        self._validate(load_spec, [])
        return load_spec

    def _check_limits(self, args: Any) -> None:
        """Refuse a request that asks for more than the server settings allow, before the job enters the queue."""
        settings = self.settings
        width, height = getattr(args, "width", None), getattr(args, "height", None)
        if isinstance(width, int) and isinstance(height, int) and width * height > settings.max_pixels:
            raise unprocessable(
                "limit_exceeded",
                f"{width}x{height} is more than limits.max_pixels ({settings.max_pixels}).",
                limit="max_pixels",
            )
        steps = getattr(args, "steps", None)
        if isinstance(steps, int) and steps > settings.max_steps:
            raise unprocessable(
                "limit_exceeded",
                f"{steps} steps is more than limits.max_steps ({settings.max_steps}).",
                limit="max_steps",
            )
        seeds = getattr(args, "seed", None) or []
        if len(seeds) > settings.max_seeds:
            raise unprocessable(
                "limit_exceeded",
                f"{len(seeds)} seeds is more than limits.max_seeds ({settings.max_seeds}).",
                limit="max_seeds",
            )

    def _check_quantize(self, quantize: Any) -> None:
        choices = self.settings.quantize_choices()
        if quantize not in choices:
            allowed = self.settings.allowed_quantize
            label = "bf16 (no quantization)" if quantize is None else f"q{quantize}"
            raise unprocessable("option_invalid", f"{label} is not in mflux_config.allowed_quantize ({allowed}).")

    def _hub_mflux_model(self, model: str) -> tuple[Path, str | None] | None:
        """(snapshot folder, source repository) when `model` is an mflux-format repository in the Hugging Face
        cache, else None. The path comes from the cache scan, never from the client."""
        if "/" not in model or model.startswith(".") or self.registry.by_repo(model):
            return None
        snapshot = hf_models.snapshot_path(model)
        if snapshot is None or not saved_models.is_saved_model(snapshot):
            return None
        return snapshot, self.catalog.quant_source(model)

    def _hub_base_model(self, model: str, source: str | None) -> str:
        entries = self.registry.by_repo(source) if source else []
        entry = next((e for e in entries if e.supported), entries[0] if entries else None)
        if entry is None:
            raise unprocessable(
                "base_model_required",
                f"The manifest does not name the base model of {model!r}. Give base_model.",
                model=model,
            )
        return entry.name

    def _saved_name(self, model: str) -> str | None:
        """The saved model a load request names, or None for a registry name."""
        if "/" in model or model.startswith("."):
            return None
        try:
            self.registry.find(model)
            return None
        except ApiError:
            pass
        candidate = self.models_dir / model
        return model if saved_models.is_saved_model(candidate) else None

    def _validate(self, load_spec: LoadSpec, generate_argv: list[str]) -> tuple[Any, list[str]]:
        """Parse and run the command's validate() step. No files are read and no weights load."""
        from mflux.utils.exceptions import ModelConfigError

        argv = load_spec.argv + (generate_argv or self._placeholder_argv(load_spec.spec))
        args, warnings = parse(load_spec.spec, argv)
        try:
            load_spec.spec.steps().validate(args)
        except ModelConfigError as exc:
            raise unprocessable("option_invalid", str(exc), model=load_spec.entry.name) from None
        return args, warnings

    @staticmethod
    def _placeholder_argv(spec: CommandSpec) -> list[str]:
        argv = ["--prompt", "x"]
        if getattr(spec.build_parser(), "require_image_paths", False):
            argv += ["--image-paths", "x.png"]
        return argv

    def submit_load(self, request: dict[str, Any]) -> Job:
        load_spec = self.build_load_spec(request)
        job = Job("load", load_spec.request, lambda job: self._do_load(job, load_spec))
        job.context["load_spec"] = load_spec
        return self.queue.submit(job, before_queue=lambda: setattr(self, "target", load_spec))

    def submit_unload(self) -> Job:
        job = Job("unload", {}, self._do_unload)
        return self.queue.submit(job, before_queue=lambda: setattr(self, "target", None))

    def _do_load(self, job: Job, load_spec: LoadSpec) -> dict[str, Any]:
        with self._job(job):
            self._unload(job)
            self._check_cached(load_spec)
            self._check_memory(load_spec)
            job.set_progress({"phase": "loading"})
            args, _ = self._validate(load_spec, [])
            model = load_spec.spec.steps().load(args)
            job.set_progress({"phase": "reading_weights"})
            materialize(model)
            model.callbacks.register(ServerCallback(self))
            self.loaded = Loaded(load_spec, model, base_tiling=getattr(model, "tiling_config", None))
            self._record_identity(load_spec)
            return self.loaded.to_dict()

    def _record_identity(self, load_spec: LoadSpec) -> None:
        """After a saved model without a sidecar loads with the client's base_model, write the sidecar, so the
        next load needs no base_model (docs/scope.md 3.10)."""
        if load_spec.saved_name is None:
            return
        path = self.models_dir / load_spec.saved_name
        if saved_models.read_sidecar(path) is None:
            quantize = saved_models.describe(path)["quantize"]
            saved_models.write_sidecar(path, load_spec.entry.name, load_spec.entry.name, quantize)

    def _do_unload(self, job: Job) -> dict[str, Any]:
        with self._job(job):
            was = self.loaded.to_dict() if self.loaded else None
            self._unload(job)
            return {"unloaded": was}

    def _unload(self, job: Job | None = None) -> None:
        if self.loaded is None:
            return
        if job is not None:
            job.set_progress({"phase": "unloading"})
        model = self.loaded.model
        self.loaded = None
        _clear_prompt_cache(model)
        del model
        free_memory()

    def _check_cached(self, load_spec: LoadSpec) -> None:
        if load_spec.saved_name is not None or load_spec.hub_repo is not None:
            return
        complete = hf_models.is_complete(load_spec.entry)
        if complete is False:
            raise not_found(
                f"{load_spec.entry.hf_repo} is not complete in the Hugging Face cache. Send a download request first.",
                model=load_spec.entry.name,
            )

    def _check_memory(self, load_spec: LoadSpec) -> None:
        need = self._estimate_bytes(load_spec)
        total = system_memory()
        if need is None or total is None:
            return
        limit = int(self.settings.memory_limit_gb * GB) if self.settings.memory_limit_gb else total
        available = min(limit, int(total * (1 - MEMORY_HEADROOM)))
        if need > available:
            raise ApiError(
                "insufficient_memory",
                507,
                f"{load_spec.entry.name} needs about {need / GB:.1f} GB, but only {available / GB:.1f} GB is available. "
                "Use a smaller quantization or a saved quantized model.",
                {"needed_gb": round(need / GB, 1), "available_gb": round(available / GB, 1)},
            )

    def _estimate_bytes(self, load_spec: LoadSpec) -> int | None:
        """The weight size on disk, scaled for quantization. A rough guide, not an exact figure."""
        if load_spec.model_path is not None:
            return saved_models._size(load_spec.model_path)
        if load_spec.saved_name is not None:
            return saved_models._size(self.models_dir / load_spec.saved_name)
        for model in hf_models.list_models(self.registry):
            if model["model_name"] == load_spec.entry.name:
                size = int(model["size_gb"] * GB)
                return size * load_spec.quantize // 16 if load_spec.quantize else size
        return None

    # -- generate ------------------------------------------------------------------------------------------------

    def submit_generate(self, request: dict[str, Any]) -> Job:
        target = self.target
        if target is None:
            raise conflict("model_not_loaded", "No model is loaded. Send POST /v1/model/load first.")
        if not isinstance(request, dict):
            raise invalid_request("The generate request must be a JSON object.")
        embed = {}
        for name in SERVER_OPTIONS:
            if request.get(name) is not None and not isinstance(request[name], bool):
                raise unprocessable("option_invalid", f"The option {name!r} must be a boolean.", option=name)
            embed[name] = request[name] if request.get(name) is not None else getattr(self.settings, name)
        request = check_image_paths(request, self.input_dir, self.resolve_input)
        mflux_request = {name: value for name, value in request.items() if name not in SERVER_OPTIONS}
        argv = to_argv(target.spec.build_parser(), mflux_request, allowed=generate_names(target.spec))
        args, warnings = self._validate(target, argv)
        self._check_limits(args)
        job = Job("generate", request, lambda job: self._do_generate(job, argv, **embed))
        job.warnings = warnings

        def check_target() -> None:
            if self.target is not target:
                raise conflict("model_changed", "The target model changed. Send the request again.")

        return self.queue.submit(job, before_queue=check_target)

    def _do_generate(
        self,
        job: Job,
        generate_argv: list[str],
        exif_enabled: bool = True,
        iptc_enabled: bool = True,
        json_sidecar_enabled: bool = True,
    ) -> dict[str, Any]:
        from mflux.callbacks.callback_manager import CallbackManager
        from mflux.utils.exceptions import StopImageGenerationException

        with self._job(job):
            if self.loaded is None:
                raise conflict("model_not_loaded", "No model is loaded. The load job before this job failed.")
            if self.loaded.state == "needs_reload":
                self._reload(job)
            loaded = self.loaded
            assert loaded is not None
            args, _ = self._validate(loaded.spec, generate_argv)
            model = loaded.model
            model.tiling_config = loaded.base_tiling
            CallbackManager._apply_vae_tiling(args, model)
            memory_saver = None
            if loaded.spec.low_ram:
                # Low-RAM mode frees model parts during the run, so the next job loads the model again.
                memory_saver = CallbackManager._register_memory_saver(args, model)
            images: list[dict[str, Any]] = []
            job.result = {"images": images}
            output_dir = self.output_dir
            try:
                for index, seed in enumerate(args.seed):
                    job.check_cancel()
                    self.image_position = (index, len(args.seed))
                    started = time.monotonic()
                    try:
                        image = loaded.spec.spec.steps().generate(model, args, seed, args.prompt)
                    except StopImageGenerationException:
                        if job.cancelled:
                            raise JobCancelled() from None
                        raise
                    path = output_dir / f"{job.id}_{seed}.png"
                    save_image(image, path, exif_enabled, iptc_enabled, json_sidecar_enabled)
                    width, height = image.image.size
                    item = {
                        "index": index,
                        "seed": seed,
                        "path": str(path),
                        "url": f"/v1/images/{job.id}/{index}",
                        "width": width,
                        "height": height,
                        "duration_seconds": round(time.monotonic() - started, 1),
                    }
                    images.append(item)
                    job.emit("image", item)
            finally:
                if memory_saver is not None:
                    _unregister(model, memory_saver)
                    loaded.state = "needs_reload"
                    # The low-RAM memory saver sets a process-wide 1 GB cache limit. Restore the server's limits.
                    self.apply_memory_limits()
                if len(getattr(model, "prompt_cache", ()) or ()) > PROMPT_CACHE_LIMIT:
                    _clear_prompt_cache(model)
            return job.result

    def _reload(self, job: Job) -> None:
        assert self.loaded is not None
        load_spec = self.loaded.spec
        self._unload(job)
        job.set_progress({"phase": "loading"})
        args, _ = self._validate(load_spec, [])
        model = load_spec.spec.steps().load(args)
        materialize(model)
        model.callbacks.register(ServerCallback(self))
        self.loaded = Loaded(load_spec, model, base_tiling=getattr(model, "tiling_config", None))

    def image_path(self, job_id: str, index: int) -> Path:
        job = self.queue.get(job_id)
        images = (job.result or {}).get("images", []) if job.type == "generate" else []
        if not 0 <= index < len(images):
            raise not_found(f"The job {job_id!r} has no image {index}.", job=job_id, index=index)
        return Path(images[index]["path"])

    # -- uploads -------------------------------------------------------------------------------------------------

    def resolve_input(self, value: str) -> Path:
        """An upload ID or a path inside the input directory, as a path to a file that exists."""
        if value.startswith("upl_"):
            matches = sorted(self.input_dir.glob(f"{safe_name(value)}.*"))
            if not matches:
                raise not_found(f"No upload with ID {value!r}.", upload=value)
            return matches[0]
        path = resolve_inside(self.input_dir, value)
        if not path.is_file():
            raise not_found(f"The input image {value!r} does not exist.", path=value)
        return path

    # -- download, save and delete --------------------------------------------------------------------------------

    def submit_download(self, model: str, confirm: bool = False) -> Job:
        if not self.settings.hf_allow_access:
            raise disabled("huggingface_models.allow_hf_access is false.")
        entry = self.registry.supported(model)
        with_weights = self.settings.hf_allow_model_download
        if with_weights and self.settings.hf_require_model_confirmation and not confirm:
            size = hf_models.estimate_bytes(entry)
            amount = f"{size / GB:.1f} GB of" if size else "an unknown amount of"
            raise conflict(
                "confirm_required",
                f"You are about to download {amount} model data from Hugging Face. Set confirm to true to continue.",
                model=entry.name,
            )
        cached = hf_models.list_models(self.registry)
        if not any(item["model_name"] == entry.name for item in cached):
            _check_limits("huggingface_models", cached, self.settings.hf_max_models, self.settings.hf_max_size_gb)
        job = Job("download", {"model": entry.name}, lambda job: self._do_download(job, entry, with_weights))
        return self.queue.submit(job)

    def _do_download(self, job: Job, entry: ModelEntry, with_weights: bool) -> dict[str, Any]:
        with self._job(job):
            result = hf_models.download(self.registry, entry, job, with_weights)
        self.rescan("hf_cache")
        return result

    def submit_save(self, request: dict[str, Any]) -> Job:
        model = request.get("model")
        quantize = request.get("quantize")
        if not isinstance(model, str):
            raise invalid_request("The save request needs 'model'.")
        from .config import QUANTIZE_CHOICES

        if quantize not in QUANTIZE_CHOICES:
            raise invalid_request(f"The save request needs 'quantize', one of {QUANTIZE_CHOICES}.")
        self._check_quantize(quantize)
        saved = saved_models.list_models(self.models_dir)
        _check_limits("mflux_models", saved, self.settings.mflux_max_models, self.settings.mflux_max_size_gb)
        entry = self.registry.supported(model)
        name = safe_name(request["name"] if "name" in request else f"{entry.name}-q{quantize}")
        if (self.models_dir / name).exists():
            raise conflict("name_in_use", f"The model directory already has {name!r}.", name=name)
        job = Job("save", {"model": entry.name, "quantize": quantize, "name": name},
                  lambda job: self._do_save(job, entry, quantize, name))  # fmt: skip
        # The save job unloads the model, so later generate requests must see no target model.
        return self.queue.submit(job, before_queue=lambda: setattr(self, "target", None))

    def _do_save(self, job: Job, entry: ModelEntry, quantize: int, name: str) -> dict[str, Any]:
        with self._job(job):
            final = self.models_dir / name
            if final.exists():
                raise conflict("name_in_use", f"The model directory already has {name!r}.", name=name)
            partial = self.models_dir / f".{name}.partial"
            shutil.rmtree(partial, ignore_errors=True)  # left by a save that failed before
            self._unload(job)  # one model in memory at a time
            job.set_progress({"phase": "loading"})
            try:
                model = build_model_for_save(entry.name, quantize)
                try:
                    job.set_progress({"phase": "saving"})
                    model.save_model(str(partial))
                finally:
                    del model
                    free_memory()
                saved_models.write_sidecar(partial, model_name=entry.name, base_model=entry.name, quantize=quantize)
                partial.rename(final)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
            result = saved_models.describe(final)
        self.rescan("mflux_dir")
        return result

    def delete_hf_model(self, repo_id: str, confirm: bool) -> dict[str, Any]:
        if not self.settings.hf_allow_delete:
            raise disabled("huggingface_models.allow_delete is false.")
        if not confirm:
            raise conflict("confirm_required", "Set confirm to true to delete the model.")
        names, _ = self._models_in_use()
        if repo_id.lower() in self._hub_repos_in_use():
            raise conflict("model_in_use", f"{repo_id} is loaded or a job in the queue uses it.", repo_id=repo_id)
        for entry in self.registry.by_repo(repo_id):
            if entry.name in names:
                raise conflict(
                    "model_in_use", f"{entry.name} is loaded or a job in the queue uses it.", model=entry.name
                )
        result = hf_models.delete(self.registry, repo_id)
        self.rescan("hf_cache")
        return result

    def delete_saved_model(self, name: str, confirm: bool) -> dict[str, Any]:
        if not self.settings.mflux_allow_delete:
            raise disabled("mflux_models.allow_delete is false.")
        if not confirm:
            raise conflict("confirm_required", "Set confirm to true to delete the model.")
        _, saved = self._models_in_use()
        result = saved_models.delete(self.models_dir, name, saved)
        self.rescan("mflux_dir")
        return result

    def delete_model(self, store: str, name: str, confirm: bool) -> dict[str, Any]:
        if store == "hf_cache":
            return self.delete_hf_model(name, confirm)
        if store == "mflux_dir":
            return self.delete_saved_model(name, confirm)
        raise invalid_request("Only a model in the store 'hf_cache' or 'mflux_dir' can be deleted.", store=store)

    def set_base_model(self, name: str, base_model: str) -> dict[str, Any]:
        result = saved_models.set_base_model(self.registry, self.models_dir, name, base_model)
        self.rescan("mflux_dir")
        return result

    def rescan(self, store: str) -> None:
        """Refresh the catalog rows of one store after a change. A failed scan must not fail the job."""
        try:
            self.catalog.scan(store)
        except Exception:
            logger.exception("catalog: the scan of %s failed", store)

    # -- helpers -------------------------------------------------------------------------------------------------

    @property
    def models_dir(self) -> Path:
        return self.settings.path("model_dir")

    @property
    def input_dir(self) -> Path:
        return self.settings.path("input_dir")

    @property
    def output_dir(self) -> Path:
        path = self.settings.path("output_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _job(self, job: Job) -> "_CurrentJob":
        return _CurrentJob(self, job)

    def resync_target(self) -> None:
        """After a job: when no load or unload job waits, the target is the model that is really loaded."""

        def resync(waiting: list[Job]) -> None:
            if not any(job.type in {"load", "unload"} for job in waiting):
                self.target = self.loaded.spec if self.loaded else None

        # Under the queue lock: submit_load and submit_unload set the target under the same lock.
        self.queue.with_waiting(resync)


class _CurrentJob:
    def __init__(self, engine: Engine, job: Job) -> None:
        self._engine = engine
        self._job = job

    def __enter__(self) -> None:
        self._engine.current_job = self._job

    def __exit__(self, *exc: Any) -> None:
        self._engine.current_job = None
        self._engine.image_position = (0, 0)
        self._engine.resync_target()


def disabled(message: str) -> ApiError:
    return ApiError("operation_disabled", 403, f"The config does not allow this operation: {message}")


def _check_limits(
    section: str, models: list[dict[str, Any]], max_models: int | None, max_size_gb: float | None
) -> None:
    """Refuse a new model when the directory already holds max_models models or max_size_gb gigabytes."""
    if max_models is not None and len(models) >= max_models:
        raise conflict("limit_reached", f"{section}.max_models is {max_models}, and {len(models)} models are there.")
    total = sum(model.get("size_gb") or 0 for model in models)
    if max_size_gb is not None and total >= max_size_gb:
        raise conflict("limit_reached", f"{section}.max_size_gb is {max_size_gb}, and the models use {total:.1f} GB.")


def _unregister(model: Any, callback: Any) -> None:
    callbacks = model.callbacks
    for name in ("before_loop", "in_loop", "after_loop", "interrupt"):
        registered = getattr(callbacks, name, [])
        if callback in registered:
            registered.remove(callback)


def save_image(image: Any, path: Path, exif: bool, iptc: bool, json_sidecar: bool = True) -> None:
    """Save a generated PNG with the metadata that the caller wants. mflux embeds EXIF, XMP and IPTC as one switch
    (ImageUtil.embed_metadata_enabled), so the switch goes off for the save, and the two parts go in one by one."""
    from mflux.utils.image_util import ImageUtil
    from mflux.utils.metadata_builder import MetadataBuilder

    metadata = image._get_metadata()
    previous = ImageUtil.embed_metadata_enabled
    ImageUtil.embed_metadata_enabled = False
    try:
        image.save(path=str(path), export_json_metadata=json_sidecar, overwrite=True)
    finally:
        ImageUtil.embed_metadata_enabled = previous
    try:
        if exif:
            ImageUtil._embed_metadata(metadata, path)
        if iptc:
            MetadataBuilder.embed_metadata(metadata, path)
    except Exception:  # noqa: BLE001 -- the image is saved; a metadata failure is a log line, as in mflux
        logger.exception("The metadata of %s was not written.", path)


def _clear_prompt_cache(model: Any) -> None:
    cache = getattr(model, "prompt_cache", None)
    if cache is not None and hasattr(cache, "clear"):
        cache.clear()


def materialize(model: Any) -> None:
    """Read all weights into memory now. MLX arrays are lazy: without this, a load job ends at once and the
    weights load during the first denoising step, so load time, memory use and an out-of-memory failure all
    show up in the first generate job instead of the load job."""
    import mlx.core as mx
    import mlx.nn as nn

    modules = (
        [model]
        if isinstance(model, nn.Module)
        else [v for v in getattr(model, "__dict__", {}).values() if isinstance(v, nn.Module)]
    )
    for module in modules:
        mx.eval(module.parameters())


def free_memory() -> None:
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def system_memory() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


# The function below follows mflux's mflux-save main() (src/mflux/models/common/cli/save.py) from step 1 on,
# statement for statement, with the parsed command line replaced by its two inputs. mflux has no save step a
# downstream project can call. tests/test_mflux_drift.py fails when mflux changes main(). See NOTICE.
def build_model_for_save(model_name: str, quantize: int) -> Any:
    from mflux.cli.parser.parsers import CommandLineParser
    from mflux.models.common.cli.save import MODEL_CLASSES, SaveDispatch
    from mflux.models.common.config import ModelConfig
    from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition

    parser = CommandLineParser(description="mlx-imagine save")

    def _error(message: str) -> None:
        raise unprocessable("option_invalid", message, model=model_name)

    setattr(parser, "error", _error)
    base_model = None
    model_path = None

    key = SaveDispatch.resolve_registry_key(parser, model_name, base_model)
    model_class = MODEL_CLASSES[key]

    if key == "ideogram-4-fp8":
        model_config = Ideogram4WeightDefinition.resolve_inference_config(model_name, base_model, model_path)
        model_path = None if Ideogram4WeightDefinition.is_builtin_name(model_name) else model_path
    else:
        model_config = ModelConfig.from_name(model_name, base_model=base_model)

    # mflux-save passes the LoRA options here. LoRA is in stage 2, so the server passes none.
    return model_class(quantize=quantize, model_path=model_path, model_config=model_config)
