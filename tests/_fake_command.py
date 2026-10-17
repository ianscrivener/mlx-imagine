"""A fake mflux command at the one boundary the server owns nothing behind: the model class (weights, Metal).

The parser and validate() are mflux's real z-image-turbo ones, so every option check is real. load() builds a
FakeModel instead of ZImage, and generate() runs the model's in-loop callbacks for each step, then returns an
image that saves a small PNG and a JSON sidecar the way GeneratedImage does."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import PIL.Image
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.z_image.cli import z_image_turbo_generate
from mflux.models.z_image.cli.z_image_turbo_generate import ZImageTurboCommand
from mflux.utils.dimension_resolver import DimensionResolver

from mlx_imagine.registry import CommandSpec

build_parser = z_image_turbo_generate.build_parser
DEFAULT_MODEL = z_image_turbo_generate.DEFAULT_MODEL
IGNORED_OPTIONS = z_image_turbo_generate.IGNORED_OPTIONS


class FakeImage:
    def __init__(self, seed: int, prompt: str, width: int, height: int) -> None:
        self.seed = seed
        self.prompt = prompt
        self.image = PIL.Image.new("RGB", (width // 64 or 1, height // 64 or 1))

    def save(self, path: str, export_json_metadata: bool = False, overwrite: bool = False) -> None:
        self.image.save(path)
        if export_json_metadata:
            Path(path).with_suffix(".json").write_text(json.dumps({"seed": self.seed, "prompt": self.prompt}))


class FakeModel:
    instances: ClassVar[list["FakeModel"]] = []
    # Tests set these to hold a run on a step, or to fail one seed.
    hold_at: ClassVar[tuple[int, int] | None] = None  # (seed, step)
    held = threading.Event()
    release = threading.Event()
    fail_on_seed: ClassVar[dict[int, BaseException]] = {}

    def __init__(self, args: Any) -> None:
        self.args = args
        self.callbacks = CallbackRegistry()
        self.tiling_config = None
        self.prompt_cache: dict[str, Any] = {}
        self.generate_calls: list[dict[str, Any]] = []
        type(self).instances.append(self)


class FakeCommand:
    latent_creator = None
    validate = staticmethod(ZImageTurboCommand.validate)

    @staticmethod
    def load(args: Any) -> FakeModel:
        ZImageTurboCommand.validate(args)
        return FakeModel(args)

    @staticmethod
    def generate(model: FakeModel, args: Any, seed: int, prompt: str) -> FakeImage:
        model.generate_calls.append({"seed": seed, "prompt": prompt, "steps": args.steps})
        failure = FakeModel.fail_on_seed.get(seed)
        if failure is not None:
            raise failure
        config = SimpleNamespace(num_inference_steps=args.steps)
        model.prompt_cache[prompt] = object()
        for callback in model.callbacks.before_loop:
            callback.call_before_loop(seed=seed, prompt=prompt, latents=None, config=config)
        for t in range(args.steps):
            if FakeModel.hold_at == (seed, t):
                FakeModel.held.set()
                FakeModel.release.wait(10)
            for callback in model.callbacks.in_loop:
                callback.call_in_loop(
                    t=t, seed=seed, prompt=prompt, latents=None, config=config, time_steps=range(args.steps)
                )
        width, height = DimensionResolver.resolve(
            width=args.width, height=args.height, reference_image_path=args.image_path
        )
        return FakeImage(seed, prompt, width, height)


FAKE_SPEC = CommandSpec(
    family="z_image",
    command="mflux-generate-z-image-turbo",
    module="_fake_command",
    steps_class="FakeCommand",
    weight_definition="mflux.models.z_image.weights.z_image_weight_definition.ZImageWeightDefinition",
)
