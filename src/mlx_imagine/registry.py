"""Which mflux models the server can run, and through which mflux command steps.

The server drives a model only through the validate/load/generate steps of its mflux command. It never copies
a command's main() or builds a model class by itself (docs/scope.md 2.1).
Which registry entry belongs to which command is not written down here: each command's own validate() step
decides, the same way it decides for the command line. validate() reads no files and loads no weights."""

import contextlib
import importlib
import io
import sys
import threading
import warnings
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from .errors import not_found, unprocessable


@dataclass(frozen=True)
class CommandSpec:
    family: str
    command: str
    module: str
    steps_class: str
    weight_definition: str
    stage: int = 1
    # Other repositories the model loads from, as ("module.ATTRIBUTE", patterns). The attribute holds the repo ID.
    extra_repos: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def load_module(self) -> ModuleType:
        return importlib.import_module(self.module)

    def steps(self) -> Any:
        return getattr(self.load_module(), self.steps_class)

    def build_parser(self) -> Any:
        return self.load_module().build_parser()

    def default_model(self) -> str | None:
        module = self.load_module()
        return getattr(module, "DEFAULT_MODEL", None) or self.build_parser().get_default("model")

    def download_patterns(self, model_name: str) -> list[str]:
        module_path, class_name = self.weight_definition.rsplit(".", 1)
        definition = getattr(importlib.import_module(module_path), class_name)
        try:
            return list(definition.get_download_patterns(model_name=model_name))
        except TypeError:  # most definitions take no argument
            return list(definition.get_download_patterns())

    def extra_downloads(self) -> list[tuple[str, list[str]]]:
        downloads = []
        for attribute, patterns in self.extra_repos:
            module_path, name = attribute.rsplit(".", 1)
            downloads.append((getattr(importlib.import_module(module_path), name), list(patterns)))
        return downloads


_M = "mflux.models"

# Stage 1: the mflux commands that have command steps.
COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("z_image", "mflux-generate-z-image-turbo", f"{_M}.z_image.cli.z_image_turbo_generate",
                "ZImageTurboCommand", f"{_M}.z_image.weights.z_image_weight_definition.ZImageWeightDefinition"),
    CommandSpec("z_image", "mflux-generate-z-image", f"{_M}.z_image.cli.z_image_generate",
                "ZImageCommand", f"{_M}.z_image.weights.z_image_weight_definition.ZImageWeightDefinition"),
    CommandSpec("ernie_image", "mflux-generate-ernie-image-turbo", f"{_M}.ernie_image.cli.ernie_image_turbo_generate",
                "ErnieImageTurboCommand", f"{_M}.ernie_image.weights.ernie_weight_definition.ErnieWeightDefinition"),
    CommandSpec("ernie_image", "mflux-generate-ernie-image", f"{_M}.ernie_image.cli.ernie_image_generate",
                "ErnieImageCommand", f"{_M}.ernie_image.weights.ernie_weight_definition.ErnieWeightDefinition"),
    CommandSpec("krea2", "mflux-generate-krea2", f"{_M}.krea2.cli.krea2_generate",
                "Krea2Command", f"{_M}.krea2.weights.krea2_weight_definition.Krea2WeightDefinition"),
    CommandSpec("lens", "mflux-generate-lens", f"{_M}.lens.cli.lens_generate",
                "LensCommand", f"{_M}.lens.weights.lens_weight_definition.LensWeightDefinition",
                extra_repos=((f"{_M}.lens.variants.txt2img.lens_image.DEFAULT_ENCODER_REPO", ("*.safetensors", "*.json")),)),
    CommandSpec("boogu", "mflux-generate-boogu", f"{_M}.boogu.cli.boogu_image_generate",
                "BooguImageCommand", f"{_M}.boogu.weights.boogu_weight_definition.BooguWeightDefinition"),
    CommandSpec("ideogram4", "mflux-generate-ideogram4", f"{_M}.ideogram4.cli.ideogram4_generate",
                "Ideogram4Command", f"{_M}.ideogram4.weights.ideogram4_weight_definition.Ideogram4WeightDefinition"),
    CommandSpec("qwen", "mflux-generate-qwen", f"{_M}.qwen.cli.qwen_image_generate",
                "QwenImageCommand", f"{_M}.qwen.weights.qwen_weight_definition.QwenWeightDefinition"),
    CommandSpec("qwen", "mflux-generate-qwen-edit", f"{_M}.qwen.cli.qwen_image_edit_generate",
                "QwenImageEditCommand", f"{_M}.qwen.weights.qwen_weight_definition.QwenWeightDefinition"),
)  # fmt: skip

# Models whose mflux command has no command steps. Each adapter is a module in mlx_imagine.adapters.
_A = "mlx_imagine.adapters"
ADAPTER_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("flux3", "mflux-generate-flux3", f"{_A}.flux3", "Flux3KleinCommand",
                f"{_M}.flux3.weights.flux3_weight_definition.Flux3KleinWeightDefinition"),
    CommandSpec("flux2", "mflux-generate-flux2", f"{_A}.flux2", "Flux2KleinCommand",
                f"{_M}.flux2.weights.flux2_weight_definition.Flux2KleinWeightDefinition"),
)  # fmt: skip

# No command is in stage 2 now. ControlNet is a non-goal (NON-GOALS.md).
STAGE_2_COMMANDS: tuple[CommandSpec, ...] = ()

REASON_NO_STEPS = "The mflux command for this model has no command steps."
REASON_STAGE_2 = "The server supports this model from stage 2."


@dataclass(frozen=True)
class ModelEntry:
    name: str
    aliases: tuple[str, ...]
    hf_repo: str
    spec: CommandSpec | None
    modes: tuple[str, ...] = ()
    reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        return self.spec is not None and self.spec.stage == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "family": self.spec.family if self.spec else None,
            "command": self.spec.command if self.spec else None,
            "hf_repo": self.hf_repo,
            "supported": self.supported,
            "modes": list(self.modes),
            "reason": self.reason,
        }


PARSE_LOCK = threading.RLock()


def parse_command_line(parser: Any, argv: list[str]) -> Any:
    """Run an mflux parser on argv. CommandLineParser.parse_args() reads sys.argv, so sys.argv is swapped
    under a lock for the call. Raises SystemExit as argparse does; the caller turns it into an error."""
    with PARSE_LOCK:
        saved = sys.argv
        sys.argv = ["mlx-imagine", *argv]
        try:
            return parser.parse_args()
        finally:
            sys.argv = saved


def accepts(spec: CommandSpec, model_name: str) -> bool:
    """True when the command's own validate() step accepts the registry name."""
    parser = spec.build_parser()
    argv = ["--model", model_name, "--prompt", "x"]
    if _requires_image_paths(parser):
        argv += ["--image-paths", "x.png"]
    try:
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = parse_command_line(parser, argv)
            spec.steps().validate(args)
    except SystemExit:
        return False
    except Exception:  # noqa: BLE001 -- ModelConfigError and its relatives: the command refuses the name
        return False
    return True


def _requires_image_paths(parser: Any) -> bool:
    return bool(getattr(parser, "require_image_paths", False))


def modes_of(spec: CommandSpec) -> tuple[str, ...]:
    parser = spec.build_parser()
    if _requires_image_paths(parser):
        return ("edit",)
    modes = ["txt2img"]
    if getattr(parser, "supports_image_to_image", False):
        modes.append("img2img")
    return tuple(modes)


def _canonical(name: str | None, available: dict[str, Any]) -> str | None:
    """The registry key of a model name or alias."""
    if name is None or name in available:
        return name
    for key, config in available.items():
        if name in config.aliases:
            return key
    return name


class Registry:
    def __init__(
        self,
        commands: tuple[CommandSpec, ...] = COMMANDS + ADAPTER_COMMANDS,
        stage_2: tuple[CommandSpec, ...] = STAGE_2_COMMANDS,
    ):
        self._commands = commands
        self._stage_2 = stage_2
        self._entries: dict[str, ModelEntry] | None = None
        self._lock = threading.Lock()

    def entries(self) -> dict[str, ModelEntry]:
        with self._lock:
            if self._entries is None:
                self._entries = self._build()
            return self._entries

    def _build(self) -> dict[str, ModelEntry]:
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        defaults = {_canonical(spec.default_model(), AVAILABLE_MODELS): spec for spec in self._commands}
        stage_2_defaults = {_canonical(spec.default_model(), AVAILABLE_MODELS): spec for spec in self._stage_2}
        modes = {spec.command: modes_of(spec) for spec in self._commands}
        entries = {}
        for name, config in AVAILABLE_MODELS.items():
            # A command whose default model is this entry owns it, ahead of a command that also accepts it
            # (mflux-generate-z-image accepts z-image-turbo, but mflux-generate-z-image-turbo owns it).
            spec = defaults.get(name) or next((s for s in self._commands if accepts(s, name)), None)
            reason = None
            if spec is None:
                spec = stage_2_defaults.get(name) or next((s for s in self._stage_2 if accepts(s, name)), None)
                reason = REASON_STAGE_2 if spec else REASON_NO_STEPS
            entries[name] = ModelEntry(
                name=name,
                aliases=tuple(alias for alias in config.aliases if alias != name),
                hf_repo=config.model_name,
                spec=spec,
                modes=modes.get(spec.command, ()) if spec and spec.stage == 1 else (),
                reason=reason,
            )
        return entries

    def find(self, name: str) -> ModelEntry:
        """The entry for a registry name, an alias or a repository ID; not_found otherwise."""
        entries = self.entries()
        if name in entries:
            return entries[name]
        lowered = name.lower()
        for entry in entries.values():
            if lowered in (alias.lower() for alias in entry.aliases):
                return entry
        # Several entries can share one repository (a ControlNet entry names its base repo). A supported one wins.
        by_repo = sorted(self.by_repo(name), key=lambda entry: not entry.supported)
        if by_repo:
            return by_repo[0]
        raise not_found(f"{name!r} is not an mflux model name, alias or repository ID.", model=name)

    def supported(self, name: str) -> ModelEntry:
        entry = self.find(name)
        if not entry.supported:
            raise unprocessable("model_not_supported", f"{entry.name}: {entry.reason}", model=entry.name)
        return entry

    def by_repo(self, repo_id: str) -> list[ModelEntry]:
        return [entry for entry in self.entries().values() if entry.hf_repo.lower() == repo_id.lower()]
