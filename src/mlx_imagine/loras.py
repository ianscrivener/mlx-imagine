"""LoRA files: list, resolve, download and delete inside the LoRA directory (docs/api.md 4.5).

A client names a LoRA by its path inside the LoRA directory, for example `style/film.safetensors`. The server
never passes a client path or a Hugging Face name to mflux: mflux would read any file, or download one."""

import math
import re
from pathlib import Path
from typing import Any

from .errors import invalid_request, not_found
from .jobs import Job
from .paths import resolve_inside

SUFFIX = ".safetensors"
MAX_LORAS = 8
MAX_DEPTH = 3
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
MB = 1000**2


def _files(lora_dir: Path) -> list[Path]:
    if not lora_dir.is_dir():
        return []
    found = []
    for path in lora_dir.rglob(f"*{SUFFIX}"):
        relative = path.relative_to(lora_dir)
        hidden = any(part.startswith(".") for part in relative.parts)
        if path.is_file() and not path.is_symlink() and not hidden and len(relative.parts) <= MAX_DEPTH:
            found.append(path)
    return sorted(found)


def list_loras(lora_dir: Path) -> list[dict[str, Any]]:
    return [
        {"name": path.relative_to(lora_dir).as_posix(), "size_mb": round(path.stat().st_size / MB, 1)}
        for path in _files(lora_dir)
    ]


def resolve(lora_dir: Path, name: object) -> Path:
    """The path of a LoRA file that exists inside lora_dir."""
    if not isinstance(name, str) or not name.endswith(SUFFIX):
        raise invalid_request(f"A LoRA name is a path inside the LoRA directory that ends in {SUFFIX}.", name=name)
    path = resolve_inside(lora_dir, name)
    if not path.is_file():
        raise not_found(f"The LoRA {name!r} is not in the LoRA directory.", name=name)
    return path


def parse_request(lora_dir: Path, value: object) -> list[dict[str, Any]]:
    """The `loras` field of a load request as [{name, scale, path}]. An item is a name or {name, scale}."""
    if not isinstance(value, list) or not value:
        raise invalid_request("'loras' must be a list of LoRA names or {name, scale} objects.")
    if len(value) > MAX_LORAS:
        raise invalid_request(f"A load request can use at most {MAX_LORAS} LoRAs.", count=len(value))
    items = []
    for item in value:
        if isinstance(item, str):
            name, scale = item, 1.0
        elif isinstance(item, dict):
            name, scale = item.get("name"), item.get("scale", 1.0)
        else:
            raise invalid_request("Each item of 'loras' is a LoRA name or a {name, scale} object.")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
            raise invalid_request("A LoRA scale must be a number.", name=name)
        path = resolve(lora_dir, name)
        items.append({"name": path.relative_to(lora_dir.resolve()).as_posix(), "scale": float(scale), "path": path})
    return items


def check_download(repo_id: object, filename: object) -> tuple[str, str]:
    if not isinstance(repo_id, str) or not REPO_ID.match(repo_id):
        raise invalid_request("'repo_id' must be a Hugging Face repository, for example owner/name.")
    parts = filename.split("/") if isinstance(filename, str) else []
    if (
        not isinstance(filename, str)
        or not filename.endswith(SUFFIX)
        or any(p in {"", ".", ".."} or p.startswith(".") for p in parts)
    ):
        raise invalid_request(f"'filename' must be a file path in the repository that ends in {SUFFIX}.")
    return repo_id, filename


def target_name(repo_id: str, filename: str) -> str:
    return f"{repo_id.replace('/', '--')}/{filename}"


def estimate_bytes(repo_id: str, filename: str) -> int | None:
    """The size of one file, from the Hub. None when the Hub does not answer."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().get_paths_info(repo_id, [filename])
    except Exception:  # noqa: BLE001 -- no network, or the Hub refuses: the caller asks without a size
        return None
    return int(getattr(info[0], "size", 0) or 0) or None if info else None


def download(lora_dir: Path, repo_id: str, filename: str, job: Job) -> dict[str, Any]:
    import shutil

    from huggingface_hub import snapshot_download

    from .hf_models import _progress_class

    folder = lora_dir / repo_id.replace("/", "--")
    job.set_progress({"phase": "downloading", "repo_id": repo_id, "file": filename})
    snapshot_download(  # ty: ignore[no-matching-overload]
        repo_id=repo_id, allow_patterns=[filename], local_dir=folder, tqdm_class=_progress_class(job)
    )
    shutil.rmtree(folder / ".cache", ignore_errors=True)  # the download bookkeeping of huggingface_hub
    name = target_name(repo_id, filename)
    path = resolve(lora_dir, name)
    return {"name": name, "size_mb": round(path.stat().st_size / MB, 1)}


def delete(lora_dir: Path, name: str, in_use: set[str]) -> dict[str, Any]:
    from .errors import conflict

    path = resolve(lora_dir, name)
    relative = path.relative_to(lora_dir.resolve()).as_posix()
    if relative in in_use:
        raise conflict("lora_in_use", f"{relative} is loaded or a job in the queue uses it.", name=relative)
    size = path.stat().st_size
    path.unlink()
    for parent in path.parents:  # remove the folders that the delete left empty, but never the LoRA directory
        if parent == lora_dir.resolve() or any(parent.iterdir()):
            break
        parent.rmdir()
    return {"name": relative, "freed_mb": round(size / MB, 1)}
