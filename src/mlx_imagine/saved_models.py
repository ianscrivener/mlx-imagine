"""The mflux format: a folder that mflux saved, often quantized, and its identity sidecar (docs/api.md 4.4).

mflux writes model.safetensors.index.json for each saved component, with the quantization level and the mflux
version, but not the model name. The server writes that name in a sidecar file, mlx-imagine.json
(docs/scope.md 3.10)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def is_mflux_layout(path: Path) -> bool:
    """True when a folder holds an mflux-format model: mflux saved it, or the server wrote its sidecar. The
    original Hugging Face layout also has model.safetensors.index.json files, but without the mflux metadata."""
    if not is_saved_model(path):
        return False
    if (path / SIDECAR).is_file():
        return True
    metadata = _index_metadata(path)
    return "quantization_level" in metadata or "mflux_version" in metadata


def set_base_model(registry: Registry, path: Path, base_model: str) -> dict[str, Any]:
    entry = registry.find(base_model)
    write_sidecar(path, model_name=entry.name, base_model=entry.name, quantize=describe(path)["quantize"])
    return describe(path)
