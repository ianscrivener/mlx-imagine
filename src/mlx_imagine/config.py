"""Server settings: the config file, the defaults and the environment mflux reads at import.

The config file is YAML (a .toml file is read as TOML) with these sections:

    mflux_config:        default_quantize, allowed_quantize, cache_dir
    mflux_models:        local_directory, remote_url, manifest_url, cron_minutes, allow_download, allow_delete, max_models, max_size_gb
    huggingface_models:  local_directory, cron_minutes, allow_hf_access, allow_model_download, require_model_confirmation, allow_delete, max_models, max_size_gb
    image_directories:   input, output, max_upload_mb
    server:              host, port, require_api_token, api_token, use_tls
    queue:               queue_max_length, job_history
    (top level)          memory_limit_gb, cache_limit_gb

The server keeps the settings as one flat Settings object. FIELDS maps each file key to its Settings field.
A key the server does not know yet is kept in the file and logged, not refused.

This module must not import mflux. mflux and huggingface_hub read MFLUX_CACHE_DIR and HF_HUB_CACHE once, at
import, so apply_environment() runs before anything imports them (cli.py makes sure of that)."""

import copy
import dataclasses
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
import yaml

from .errors import invalid_request

logger = logging.getLogger(__name__)

CONFIG_ENV = "MLX_IMAGINE_CONFIG"
DEFAULT_CONFIG_PATH = Path("~/.config/mlx-imagine/config.yaml")
YAML_SUFFIXES = frozenset({".yaml", ".yml"})
QUANTIZE_CHOICES = (3, 4, 5, 6, 8)
# "bf16" in allowed_quantize allows a load without quantization.
UNQUANTIZED = "bf16"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass
class Settings:
    # mflux_config
    default_quantize: int | None = None
    allowed_quantize: str = "q3 q4 q5 q6 q8 bf16"
    mflux_cache_dir: str | None = None
    # mflux_models
    model_dir: str = "~/mlx-imagine/models"
    mflux_remote_url: str = "https://huggingface.co/mflux-community"
    mflux_manifest_url: str = (
        "https://huggingface.co/buckets/mflux-community/ci/resolve/models_community_manifest.jsonl"
    )
    mflux_cron_minutes: int = 5
    mflux_allow_download: bool = True
    mflux_allow_delete: bool = True
    mflux_max_models: int | None = None
    mflux_max_size_gb: float | None = None
    # huggingface_models
    hf_hub_cache: str | None = None
    hf_cron_minutes: int = 5
    hf_allow_access: bool = True
    hf_allow_model_download: bool = True
    hf_require_model_confirmation: bool = False
    hf_allow_delete: bool = True
    hf_max_models: int | None = None
    hf_max_size_gb: float | None = None
    # image_directories
    input_dir: str = "~/mlx-imagine/input"
    output_dir: str = "~/mlx-imagine/output"
    max_upload_mb: int = 50
    # server
    host: str = "127.0.0.1"
    port: int = 8765
    require_api_token: bool = False
    api_token: str | None = None
    use_tls: bool = False
    # services
    api_enable: bool = True
    mcp_enable: bool = True
    swagger_enable: bool = True
    # queue
    queue_max_length: int = 100
    job_history: int = 200
    # top level
    data_dir: str = "~/mlx-imagine/data"
    memory_limit_gb: float | None = None
    # MLX keeps freed buffers in a cache for reuse. Without a limit, one 512x512 Z-Image Turbo image left 14.8 GB
    # in it. 4 GB keeps the speed benefit for the next job and gives the rest back to the system.
    cache_limit_gb: float | None = 4.0

    def path(self, name: str) -> Path:
        value = getattr(self, name)
        return Path(value).expanduser().resolve()

    @property
    def token(self) -> str | None:
        """The token clients must send, or None when the server does not check one."""
        return self.api_token if self.require_api_token else None

    def quantize_choices(self) -> set[int | None]:
        """The allowed quantizations. None stands for bf16 (no quantization)."""
        return parse_allowed_quantize(self.allowed_quantize)

    def public_dict(self) -> dict[str, Any]:
        data = to_file_dict(self)
        if data["server"].get("api_token") is not None:
            data["server"]["api_token"] = "***"
        return data


# (section, key) -> (Settings field, accepted types). A section of None is the top level of the file.
_OPTIONAL_NUMBER = (int, float, type(None))
FIELDS: dict[tuple[str | None, str], tuple[str, tuple[type, ...]]] = {
    ("mflux_config", "default_quantize"): ("default_quantize", (int, type(None))),
    ("mflux_config", "allowed_quantize"): ("allowed_quantize", (str, list)),
    ("mflux_config", "cache_dir"): ("mflux_cache_dir", (str, type(None))),
    ("mflux_models", "local_directory"): ("model_dir", (str,)),
    ("mflux_models", "remote_url"): ("mflux_remote_url", (str,)),
    ("mflux_models", "manifest_url"): ("mflux_manifest_url", (str,)),
    ("mflux_models", "cron_minutes"): ("mflux_cron_minutes", (int,)),
    ("mflux_models", "allow_download"): ("mflux_allow_download", (bool,)),
    ("mflux_models", "allow_delete"): ("mflux_allow_delete", (bool,)),
    ("mflux_models", "max_models"): ("mflux_max_models", (int, type(None))),
    ("mflux_models", "max_size_gb"): ("mflux_max_size_gb", _OPTIONAL_NUMBER),
    ("huggingface_models", "local_directory"): ("hf_hub_cache", (str, type(None))),
    ("huggingface_models", "cron_minutes"): ("hf_cron_minutes", (int,)),
    ("huggingface_models", "allow_hf_access"): ("hf_allow_access", (bool,)),
    ("huggingface_models", "allow_model_download"): ("hf_allow_model_download", (bool,)),
    ("huggingface_models", "require_model_confirmation"): ("hf_require_model_confirmation", (bool,)),
    ("huggingface_models", "allow_delete"): ("hf_allow_delete", (bool,)),
    ("huggingface_models", "max_models"): ("hf_max_models", (int, type(None))),
    ("huggingface_models", "max_size_gb"): ("hf_max_size_gb", _OPTIONAL_NUMBER),
    ("image_directories", "input"): ("input_dir", (str,)),
    ("image_directories", "output"): ("output_dir", (str,)),
    ("image_directories", "max_upload_mb"): ("max_upload_mb", (int,)),
    ("server", "host"): ("host", (str,)),
    ("server", "port"): ("port", (int,)),
    ("server", "require_api_token"): ("require_api_token", (bool,)),
    ("server", "api_token"): ("api_token", (str, type(None))),
    ("server", "use_tls"): ("use_tls", (bool,)),
    ("services", "api_enable"): ("api_enable", (bool,)),
    ("services", "mcp_enable"): ("mcp_enable", (bool,)),
    ("services", "swagger_enable"): ("swagger_enable", (bool,)),
    ("queue", "queue_max_length"): ("queue_max_length", (int,)),
    ("queue", "job_history"): ("job_history", (int,)),
    (None, "data_directory"): ("data_dir", (str,)),
    (None, "memory_limit_gb"): ("memory_limit_gb", _OPTIONAL_NUMBER),
    (None, "cache_limit_gb"): ("cache_limit_gb", _OPTIONAL_NUMBER),
}
SECTIONS = frozenset(section for section, _ in FIELDS if section is not None)
NAMES = {field: ".".join(part for part in location if part) for location, (field, _) in FIELDS.items()}
TYPES = {field: accepted for field, accepted in FIELDS.values()}

# A change to one of these takes effect only after a restart (docs/scope.md 3.9).
RESTART_FIELDS = frozenset(
    {
        "hf_hub_cache",
        "hf_allow_access",
        "api_enable",
        "mcp_enable",
        "swagger_enable",
        "mflux_cache_dir",
        "host",
        "port",
        "require_api_token",
        "api_token",
        "use_tls",
    }
)
PATH_FIELDS = frozenset({"hf_hub_cache", "mflux_cache_dir", "model_dir", "input_dir", "output_dir", "data_dir"})
POSITIVE_FIELDS = frozenset(
    {"memory_limit_gb", "cache_limit_gb", "mflux_max_size_gb", "hf_max_size_gb", "mflux_max_models", "hf_max_models"}
)
AT_LEAST_ONE_FIELDS = frozenset({"queue_max_length", "job_history", "max_upload_mb"})
NON_NEGATIVE_FIELDS = frozenset({"mflux_cron_minutes", "hf_cron_minutes"})


def parse_allowed_quantize(value: str | list) -> set[int | None]:
    tokens = value.split() if isinstance(value, str) else [str(item) for item in value]
    choices: set[int | None] = set()
    for token in tokens:
        token = token.strip().lower()
        match = re.fullmatch(r"q?(\d+)", token)
        if token == UNQUANTIZED:
            choices.add(None)
        elif match and int(match.group(1)) in QUANTIZE_CHOICES:
            choices.add(int(match.group(1)))
        else:
            raise invalid_request(
                f"mflux_config.allowed_quantize: {token!r} is not one of q3 q4 q5 q6 q8 bf16.",
                setting="mflux_config.allowed_quantize",
            )
    return choices


def config_path() -> Path:
    return Path(os.environ.get(CONFIG_ENV, str(DEFAULT_CONFIG_PATH))).expanduser()


def _read(path: Path) -> dict[str, Any]:
    """The mapping in a YAML (.yaml, .yml) or TOML file. The suffix selects the format."""
    text = path.read_text(encoding="utf-8")
    data = (yaml.safe_load(text) or {}) if path.suffix in YAML_SUFFIXES else toml.loads(text)
    if not isinstance(data, dict):
        raise invalid_request(f"The config file {path} must hold a mapping of settings.")
    return data


def flatten(data: dict[str, Any], strict: bool) -> dict[str, Any]:
    """Settings fields from a nested mapping. An unknown key is an error when strict, else a log line."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if key in SECTIONS:
            if not isinstance(value, dict):
                raise invalid_request(f"The section {key!r} must be a mapping.", setting=key)
            for inner, inner_value in value.items():
                _put(flat, (key, inner), inner_value, strict)
        else:
            _put(flat, (None, key), value, strict)
    return flat


def _put(flat: dict[str, Any], location: tuple[str | None, str], value: Any, strict: bool) -> None:
    name = ".".join(part for part in location if part)
    if location not in FIELDS:
        if strict:
            raise invalid_request(f"Unknown setting {name!r}.", setting=name)
        logger.warning("config: the server does not use the setting %r yet", name)
        return
    flat[FIELDS[location][0]] = value


def to_file_dict(settings: Settings) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for (section, key), (field, _) in FIELDS.items():
        value = getattr(settings, field)
        if section is None:
            data[key] = value
        else:
            data.setdefault(section, {})[key] = value
    return data


def load(path: Path | None = None) -> Settings:
    path = path or config_path()
    settings = Settings()
    if not path.is_file():
        return settings
    flat = flatten(_read(path), strict=False)
    # A relative directory is relative to the config file, not to the directory the server starts in.
    for name in PATH_FIELDS & flat.keys():
        value = flat[name]
        if isinstance(value, str) and not value.startswith("~") and not Path(value).is_absolute():
            flat[name] = str((path.parent / value).resolve())
    _apply(settings, flat)
    return settings


def save(settings: Settings, path: Path | None = None) -> None:
    """Write the settings. Keys the server does not know are kept. Comments are not kept."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(_read(path)) if path.is_file() else {}
    for section, values in to_file_dict(settings).items():
        if isinstance(values, dict):
            target = data.setdefault(section, {})
            for key, value in values.items():
                if value is None:
                    target.pop(key, None)
                else:
                    target[key] = value
        elif values is None:
            data.pop(section, None)
        else:
            data[section] = values
    _write(path, data)


def save_changes(changes: dict[str, Any], path: Path | None = None) -> None:
    """Write only the keys in `changes`, as the client sent them, into the config file.

    The live Settings also hold values that must not go into the file: the `serve --host/--port` overrides
    of this run, the defaults, and the absolute form of relative directories. update() checks `changes`
    before this runs."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(_read(path)) if path.is_file() else {}
    for key, value in changes.items():
        if key in SECTIONS and isinstance(value, dict):
            target = data.setdefault(key, {})
            for inner, inner_value in value.items():
                _set_or_remove(target, inner, inner_value)
        else:
            _set_or_remove(data, key, value)
    _write(path, data)


def _set_or_remove(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _write(path: Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, sort_keys=False) if path.suffix in YAML_SUFFIXES else toml.dumps(data)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)  # the file can hold the API token
    tmp.replace(path)


def update(settings: Settings, changes: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check and apply changes in the nested file layout. Returns (applied now, applied after restart), as
    "section.key" names. Raises before it changes anything if one value is not valid."""
    flat = flatten(changes, strict=True)
    candidate = dataclasses.replace(settings)
    _apply(candidate, flat)
    applied, restart = [], []
    for field in flat:
        if getattr(candidate, field) == getattr(settings, field):
            continue
        (restart if field in RESTART_FIELDS else applied).append(NAMES[field])
        setattr(settings, field, getattr(candidate, field))
    return applied, restart


def _apply(settings: Settings, flat: dict[str, Any]) -> None:
    for field, value in flat.items():
        name = NAMES[field]
        accepted = TYPES[field]
        if (isinstance(value, bool) and bool not in accepted) or not isinstance(value, accepted):
            raise invalid_request(f"Setting {name!r} has the wrong type.", setting=name)
        if field in PATH_FIELDS and value is not None:
            _check_directory(name, value)
        if field == "port" and not 1 <= value <= 65535:
            raise invalid_request("Setting 'server.port' must be from 1 to 65535.", setting=name)
        if field in POSITIVE_FIELDS and value is not None and value <= 0:
            raise invalid_request(f"Setting {name!r} must be greater than 0.", setting=name)
        if field in NON_NEGATIVE_FIELDS and value < 0:
            raise invalid_request(f"Setting {name!r} must be 0 or more.", setting=name)
        if field in AT_LEAST_ONE_FIELDS and value < 1:
            raise invalid_request(f"Setting {name!r} must be 1 or more.", setting=name)
        if field == "allowed_quantize":
            parse_allowed_quantize(value)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
        setattr(settings, field, value)
    if settings.default_quantize is not None and settings.default_quantize not in settings.quantize_choices():
        raise invalid_request(
            f"mflux_config.default_quantize {settings.default_quantize} is not in allowed_quantize.",
            setting="mflux_config.default_quantize",
        )


def _check_directory(name: str, value: str) -> None:
    path = Path(value).expanduser()
    if path.exists() and not path.is_dir():
        raise invalid_request(f"Setting {name!r}: {value!r} is not a directory.", setting=name)


def check_startup(settings: Settings) -> None:
    """Refuse a start that would not be safe, or that would not do what the config asks."""
    if settings.use_tls:
        raise invalid_request("server.use_tls is not supported yet. Put a TLS proxy in front of the server.")
    if settings.require_api_token and not settings.api_token:
        raise invalid_request("server.require_api_token is true, but server.api_token is not set.")
    if settings.host not in LOCAL_HOSTS and not settings.token:
        raise invalid_request(
            f"Binding to {settings.host} requires server.require_api_token: true and server.api_token."
        )


def apply_environment(settings: Settings) -> None:
    """Set the environment that mflux and huggingface_hub read at import. Call before importing them."""
    if settings.hf_hub_cache:
        # huggingface_models.local_directory holds the models--<owner>--<name> folders: it is the hub cache itself.
        os.environ["HF_HUB_CACHE"] = str(Path(settings.hf_hub_cache).expanduser())
    if not settings.hf_allow_access:
        os.environ["HF_HUB_OFFLINE"] = "1"
    if settings.mflux_cache_dir:
        os.environ["MFLUX_CACHE_DIR"] = str(Path(settings.mflux_cache_dir).expanduser())


def make_directories(settings: Settings) -> None:
    for name in ("model_dir", "input_dir", "output_dir", "data_dir"):
        settings.path(name).mkdir(parents=True, exist_ok=True)
