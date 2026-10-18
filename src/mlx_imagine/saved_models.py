"""Models that mflux saved to the model directory: list, identity sidecar and delete (docs/api.md 4.4).

mflux writes model.safetensors.index.json for each saved component, with the quantization level and the mflux
version, but not the model name. The server writes that name in a sidecar file, mlx-imagine.json
(docs/scope.md 3.10)."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import conflict, not_found, unprocessable
from .paths import resolve_inside, safe_name
from .registry import Registry

SIDECAR = "mlx-imagine.json"
INDEX = "model.safetensors.index.json"
GB = 1000**3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def is_saved_model(path: Path) -> bool:
    return path.is_dir() and (any(path.glob(f"*/{INDEX}")) or (path / INDEX).is_file())


def _index_metadata(path: Path) -> dict[str, Any]:
    for index in sorted([*path.glob(f"*/{INDEX}"), *path.glob(INDEX)]):
        try:
            return json.loads(index.read_text(encoding="utf-8")).get("metadata", {})
        except (OSError, ValueError):  # noqa: PERF203 -- one broken index must not hide the next
            continue
    return {}


def read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((path / SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_sidecar(path: Path, model_name: str, base_model: str, quantize: int | None) -> None:
    data = {"model_name": model_name, "base_model": base_model, "quantize": quantize, "saved_at": _now()}
    (path / SIDECAR).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def describe(path: Path) -> dict[str, Any]:
    metadata = _index_metadata(path)
    sidecar = read_sidecar(path)
    level = metadata.get("quantization_level")
    quantize = int(level) if level not in (None, "None") and str(level).isdigit() else None
    return {
        "name": path.name,
        "path": str(path),
        "model_name": sidecar.get("model_name") if sidecar else None,
        "base_model": sidecar.get("base_model") if sidecar else None,
        "quantize": quantize,
        "mflux_version": metadata.get("mflux_version"),
        "saved_at": sidecar.get("saved_at") if sidecar else None,
        "size_gb": round(_size(path) / GB, 2),
        "has_sidecar": sidecar is not None,
    }


def list_models(model_dir: Path) -> list[dict[str, Any]]:
    if not model_dir.is_dir():
        return []
    # A name that starts with "." is a save in progress, or one that failed. It is not a saved model.
    return [
        describe(path) for path in sorted(model_dir.iterdir()) if not path.name.startswith(".") and is_saved_model(path)
    ]


def find(model_dir: Path, name: str) -> Path:
    path = resolve_inside(model_dir, safe_name(name))
    if not is_saved_model(path):
        raise not_found(f"No saved model named {name!r} in the model directory.", name=name)
    return path


def set_base_model(registry: Registry, model_dir: Path, name: str, base_model: str) -> dict[str, Any]:
    path = find(model_dir, name)
    entry = registry.find(base_model)
    quantize = describe(path)["quantize"]
    write_sidecar(path, model_name=entry.name, base_model=entry.name, quantize=quantize)
    return describe(path)


def base_model_of(model_dir: Path, name: str, given: str | None) -> str:
    """The registry name a saved model loads as: the client's base_model, or the sidecar's."""
    if given:
        return given
    sidecar = read_sidecar(find(model_dir, name))
    if not sidecar or not sidecar.get("base_model"):
        raise unprocessable(
            "base_model_required",
            f"The saved model {name!r} has no {SIDECAR}. Give base_model, or set it with PATCH /v8/models/mflux_dir/{name}.",
            name=name,
        )
    return sidecar["base_model"]


def delete(model_dir: Path, name: str, in_use: set[str]) -> dict[str, Any]:
    path = find(model_dir, name)
    if path.name in in_use:
        raise conflict("model_in_use", f"The saved model {name!r} is loaded or a job in the queue uses it.", name=name)
    freed = _size(path)
    shutil.rmtree(path)
    return {"name": name, "freed_gb": round(freed / GB, 2)}
