"""Requests use the names of the mflux command options (docs/scope.md 3.6, option C).

A request body is a JSON object whose keys are argparse destination names (`image_path` for --image-path).
The server turns it into a command line and runs the command's own build_parser() on it, so every check and
default the command line has applies. The JSON Schemas come from the same parser, through mflux-capabilities'
describe_command(), so the server never keeps its own copy of an option list."""

import argparse
import contextlib
import io
import warnings
from pathlib import Path
from typing import Any

from .errors import ApiError, invalid_request, unprocessable
from .registry import CommandSpec, parse_command_line

# Options that change which model loads, or how. They belong to the load request.
LOAD_OPTIONS = frozenset({"model", "base_model", "quantize", "low_ram", "float32", "compute_precision"})

LORAS_SCHEMA = {
    "type": "array",
    "items": {
        "anyOf": [
            {"type": "string"},
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "scale": {"type": "number", "default": 1.0}},
                "required": ["name"],
            },
        ]
    },
    "description": "LoRA files in the LoRA directory: a name, or an object with a name and a scale. See GET /v1/loras.",
}

# Generate options that the server handles itself. They are not mflux options. A request value replaces the
# image_metadata setting for that request.
SERVER_OPTIONS = {
    "exif_enabled": {
        "type": "boolean",
        "description": "Embed the generation data in the EXIF UserComment of each PNG. The default is the setting.",
    },
    "iptc_enabled": {
        "type": "boolean",
        "description": "Embed the IPTC and XMP data (prompt, seed, model) in each PNG. The default is the setting.",
    },
    "json_sidecar_enabled": {
        "type": "boolean",
        "description": "Write a .metadata.json file next to each PNG. The default is the setting.",
    },
}

# Generate options that name a folder inside the configured input or output directory (docs/api.md 5.2).
SUBDIR_OPTIONS = {
    "input_subdir": {
        "type": "string",
        "description": "A folder inside the input directory. A relative image_path is read from it. It must exist.",
    },
    "output_subdir": {
        "type": "string",
        "description": "A folder inside the output directory for the images of this request. The server makes it.",
    },
}

# Options the server does not accept (docs/scope.md 3.6 and docs/api.md 5.2). Each one writes files the
# client names, reads files outside the input directory, changes process-wide state, or is in stage 2.
BLOCKED_OPTIONS = frozenset(
    {
        "help",
        "verbose",
        "output",
        "prompt_file",
        "config_from_metadata",
        "stepwise_image_output_dir",
        "battery_percentage_stop_limit",
        "mlx_cache_limit_gb",
        "metadata",
        "no_metadata",
        "model_path",
        "image",
        # LoRA: the parser resolves LoRA names when it parses, and that can download. A load request uses 'loras'.
        "lora",
        "lora_paths",
        "lora_scales",
        "lora_style",
        "bake_lora",
    }
)

# What a client reads in the schema for an option that names an image. These replace the help text of mflux, which
# talks about local paths. A client sends an upload ID or a path inside the input directory.
_WHERE = "An upload ID (from upload_image or POST /v1/uploads), or a path inside the input directory."
IMAGE_DESCRIPTIONS = {
    "image_path": f"The input image. {_WHERE}",
    "image_paths": f"The input images. Each one: {_WHERE.lower()}",
    "controlnet_image_path": (
        "A control image that is ready to use: canny edges, a depth map, a pose image, and so on. "
        f"The server does not make it. {_WHERE}"
    ),
    "mask_image": (
        "The mask for image_path, with the same size. White is drawn again, black stays. "
        f"Check it with preview_mask. {_WHERE}"
    ),
    "control": (
        "A list of controls, each 'type:image[:strength]'. The type is canny, depth, hed, mlsd or pose. "
        "The image is a ready control image of that type. The server does not make it. Give an upload ID, or a "
        "path inside the input directory. The strength is 0.65 to 1.0. Example: 'canny:upl_1a2b3c:0.8'."
    ),
}

# Options that name an input image. Each value must resolve inside the input directory.
IMAGE_OPTIONS = frozenset({"image_path", "image_paths", "controlnet_image_path", "mask_image"})

_JSON_TYPES = {
    "bool": {"type": "boolean"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "str": {"type": "string"},
    "path": {"type": "string"},
    "int-or-scale": {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": "^[0-9.]+x$"}]},
}


class OptionError(ApiError):
    pass


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions if action.option_strings}


def to_argv(
    parser: argparse.ArgumentParser, request: dict[str, Any], allowed: frozenset[str] | None = None
) -> list[str]:
    """The command line for a request. Refuses unknown and blocked names before the parser runs."""
    actions = _actions(parser)
    argv: list[str] = []
    for name, value in request.items():
        if name in BLOCKED_OPTIONS:
            hint = " Use 'loras' in the load request." if name.startswith(("lora", "bake_lora")) else ""
            raise unprocessable("option_blocked", f"The server does not accept the option {name!r}.{hint}", option=name)
        if name not in actions:
            raise unprocessable("option_invalid", f"Unknown option {name!r} for this model.", option=name)
        if allowed is not None and name not in allowed:
            code = "option_for_load" if name in LOAD_OPTIONS else "option_invalid"
            raise unprocessable(code, f"The option {name!r} is not allowed in this request.", option=name)
        if value is None:
            continue
        argv += _tokens(actions[name], name, value)
    return argv


def _tokens(action: argparse.Action, name: str, value: Any) -> list[str]:
    flag = next((o for o in action.option_strings if o.startswith("--")), action.option_strings[0])
    if isinstance(action, argparse.BooleanOptionalAction):
        _require(isinstance(value, bool), name, "a boolean")
        return [flag] if value else [f"--no-{flag[2:]}"]
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        _require(isinstance(value, bool), name, "a boolean")
        store_true = isinstance(action, argparse._StoreTrueAction)
        return [flag] if value == store_true else []
    if isinstance(action, argparse._AppendAction) and action.nargs is None:  # a repeatable option, like --control
        items = value if isinstance(value, list) else [value]
        _require(bool(items) and all(isinstance(item, str) for item in items), name, "a list of strings")
        return [token for item in items for token in ([f"{flag}={item}"] if item.startswith("-") else [flag, item])]
    if action.nargs in ("+", "*") or (isinstance(action.nargs, int) and action.nargs > 1):
        items = value if isinstance(value, list) else [value]
        _require(all(_scalar(item) for item in items), name, "a list of numbers or strings")
        return [flag, *(str(item) for item in items)] if items or action.nargs == "*" else []
    _require(_scalar(value), name, "a number or a string")
    return [f"{flag}={value}"] if str(value).startswith("-") else [flag, str(value)]


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _require(condition: bool, name: str, kind: str) -> None:
    if not condition:
        raise unprocessable("option_invalid", f"The option {name!r} must be {kind}.", option=name)


def parse(spec: CommandSpec, argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Run the command's parser. Returns the namespace and the warnings the parser gave."""
    parser = spec.build_parser()
    stderr = io.StringIO()

    def _error(message: str) -> None:
        raise unprocessable("option_invalid", message, command=spec.command)

    setattr(parser, "error", _error)
    with contextlib.redirect_stderr(stderr), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            args = parse_command_line(parser, argv)
        except SystemExit:
            raise unprocessable("option_invalid", stderr.getvalue().strip() or "The options are not valid.") from None
    return args, [str(warning.message) for warning in caught]


def _resolve_control(spec: object, resolve: Any) -> str:
    """`type:path[:strength]` with the path resolved inside the input directory. The split follows mflux: a last
    part that is a number is the strength."""
    if not isinstance(spec, str) or spec.count(":") < 1:
        raise invalid_request("A control is a string like 'canny:image.png' or 'pose:upl_1:0.8'.", control=spec)
    kind, _, rest = spec.partition(":")
    path, strength = rest, None
    head, sep, tail = rest.rpartition(":")
    if sep:
        try:
            float(tail)
            path, strength = head, tail
        except ValueError:
            pass
    return ":".join([kind, str(resolve(path)), *([strength] if strength else [])])


def check_image_paths(request: dict[str, Any], input_dir: Path, resolve: Any) -> dict[str, Any]:
    """A copy of the request with each image option resolved inside input_dir (an upload ID also works)."""
    checked = dict(request)
    if request.get("control") is not None:
        specs = request["control"] if isinstance(request["control"], list) else [request["control"]]
        checked["control"] = [_resolve_control(spec, resolve) for spec in specs]
    for name in IMAGE_OPTIONS & request.keys():
        value = request[name]
        if value is None:
            continue
        if isinstance(value, list):
            if not all(isinstance(item, str) for item in value):
                raise invalid_request(f"The option {name!r} must be a list of paths or upload IDs.", option=name)
            checked[name] = [str(resolve(item)) for item in value]
        elif isinstance(value, str):
            checked[name] = str(resolve(value))
        else:
            raise invalid_request(f"The option {name!r} must be a path or an upload ID.", option=name)
    return checked


def schemas(spec: CommandSpec) -> dict[str, Any]:
    """The load and generate JSON Schemas of one command, from its parser."""
    from mflux.cli.capabilities import describe_command

    described = describe_command(spec.command, spec.module)
    parser = spec.build_parser()
    by_flag = {}
    for action in parser._actions:
        for option in action.option_strings:
            by_flag[option] = action
    load: dict[str, Any] = {}
    generate: dict[str, Any] = {}
    required: list[str] = []
    for record in described.get("options", []):
        action = by_flag[record["flag"]]
        name = action.dest
        if name in BLOCKED_OPTIONS or record.get("status") in {"ignored", "rejected"}:
            continue
        schema: dict[str, Any] = dict(_JSON_TYPES.get(record["type"], {"type": "string"}))
        if "choices" in record:
            schema = {"enum": [_choice(choice, record["type"]) for choice in record["choices"]]}
        if record.get("nargs") in {"+", "*"} or (isinstance(action, argparse._AppendAction) and action.nargs is None):
            schema = {"type": "array", "items": schema}
        if name in IMAGE_DESCRIPTIONS:
            schema["description"] = IMAGE_DESCRIPTIONS[name]
        elif record.get("help"):
            schema["description"] = record["help"].replace("%%", "%")
        default = record.get("parser_default")
        if default is not None and name != "seed":
            schema["default"] = default
        (load if name in LOAD_OPTIONS else generate)[name] = schema
        if record.get("required") and name not in LOAD_OPTIONS and name != "prompt":
            required.append(name)
    if "--lora" in by_flag:
        load["loras"] = LORAS_SCHEMA
    generate.update({name: dict(schema) for name, schema in {**SERVER_OPTIONS, **SUBDIR_OPTIONS}.items()})
    if "prompt" in generate:
        required.append("prompt")
    if getattr(parser, "require_image_paths", False):
        required.append("image_paths")
    if "seed" in generate:
        generate["seed"]["description"] = "One image for each seed. Default: one random seed."
    return {
        "load_schema": {"type": "object", "properties": load, "additionalProperties": False},
        "generate_schema": {
            "type": "object",
            "properties": generate,
            "required": required,
            "additionalProperties": False,
        },
    }


def _choice(choice: str, type_name: str) -> Any:
    if type_name == "int":
        with contextlib.suppress(ValueError):
            return int(choice)
    return choice


def ignored_options(spec: CommandSpec) -> dict[str, str]:
    """Options that the command always ignores or refuses: option name -> the reason mflux gives."""
    from mflux.cli.capabilities import describe_command

    described = describe_command(spec.command, spec.module)
    by_flag = {option: action for action in spec.build_parser()._actions for option in action.option_strings}
    return {
        by_flag[record["flag"]].dest: record.get("reason", "")
        for record in described.get("options", [])
        if record.get("status") in {"ignored", "rejected"}
    }


def conditionally_ignored(spec: CommandSpec, request: dict[str, Any], args: Any, model_config: Any) -> dict[str, str]:
    """Options of this request that the model ignores because of the effective guidance. mflux checks these on
    sys.argv, which the server cannot use, so the server checks the parsed values."""
    module = spec.load_module()
    conditional = getattr(module, "CONDITIONAL_OPTIONS", {})
    found: dict[str, str] = {}
    supports = getattr(model_config, "supports_guidance", None)
    default = getattr(module, "DEFAULT_GUIDANCE", None)
    guidance = (
        args.guidance if getattr(args, "guidance", None) is not None else (default if default is not None else 0.0)
    )
    if "--negative-prompt" in conditional and request.get("negative_prompt"):
        # krea2 builds the unconditional branch at any guidance but its default of 1.0. The others need above 1.0.
        off = guidance == 1.0 if default is not None else guidance <= 1.0
        if supports is False or off:
            found["negative_prompt"] = conditional["--negative-prompt"]["reason"]
    if "--guidance" in conditional and request.get("guidance") is not None and supports is False:
        found["guidance"] = conditional["--guidance"]["reason"]
    return found


def generate_names(spec: CommandSpec) -> frozenset[str]:
    parser = spec.build_parser()
    return frozenset(name for name in _actions(parser) if name not in LOAD_OPTIONS and name not in BLOCKED_OPTIONS)


def load_names(spec: CommandSpec) -> frozenset[str]:
    parser = spec.build_parser()
    return frozenset(name for name in _actions(parser) if name in LOAD_OPTIONS)
