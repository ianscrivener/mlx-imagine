"""The compatible models in the local Hugging Face cache: list, download and delete (docs/api.md 4.3)."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm as tqdm_auto

from .errors import ApiError, not_found
from .jobs import Job
from .registry import ModelEntry, Registry

GB = 1000**3


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cache_repos() -> dict[str, Any]:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        info = scan_cache_dir()
    except CacheNotFound:
        return {}
    return {repo.repo_id.lower(): repo for repo in info.repos if repo.repo_type == "model"}


def latest_snapshot(repo: Any) -> Path | None:
    """The folder of the newest revision of a cached repository."""
    if not repo.revisions:
        return None
    return Path(max(repo.revisions, key=lambda revision: revision.last_modified).snapshot_path)


def snapshot_path(repo_id: str) -> Path | None:
    repo = _cache_repos().get(repo_id.lower())
    return latest_snapshot(repo) if repo is not None else None


def is_complete(entry: ModelEntry) -> bool | None:
    """True when the cache holds every file the model loads. None when the server cannot tell."""
    if not entry.supported or entry.spec is None:
        return None
    from mflux.models.common.resolution.path_resolution import PathResolution

    patterns = entry.spec.download_patterns(entry.name)
    return PathResolution._find_complete_cached_snapshot(entry.hf_repo, patterns) is not None


def list_models(registry: Registry) -> list[dict[str, Any]]:
    repos = _cache_repos()
    models = []
    seen: set[str] = set()
    # Several entries can share one repository (a ControlNet entry names its base repo). List the repository one
    # time, under a supported entry when there is one, as Registry.find does.
    for entry in sorted(registry.entries().values(), key=lambda entry: not entry.supported):
        repo = repos.get(entry.hf_repo.lower())
        if repo is None or repo.repo_id.lower() in seen:
            continue
        seen.add(repo.repo_id.lower())
        models.append(
            {
                "repo_id": repo.repo_id,
                "model_name": entry.name,
                "supported": entry.supported,
                "size_gb": round(repo.size_on_disk / GB, 2),
                "revisions": sorted(revision.commit_hash[:7] for revision in repo.revisions),
                "complete": is_complete(entry),
                "last_accessed": _iso(repo.last_accessed),
            }
        )
    return sorted(models, key=lambda model: model["model_name"])


def entry_for_cache(registry: Registry, entry: ModelEntry) -> dict[str, Any]:
    for model in list_models(registry):
        if model["model_name"] == entry.name:
            return model
    raise not_found(f"{entry.hf_repo} is not in the Hugging Face cache.", model=entry.name)


def _progress_class(job: Job) -> type:
    class JobProgress(tqdm_auto):  # ty: ignore[unsupported-base]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n: float | None = 1) -> bool | None:
            job.check_cancel()
            result = super().update(n)
            if self.unit == "B":
                job.set_progress({"bytes_done": int(self.n), "bytes_total": int(self.total or 0), "file": self.desc})
            else:
                job.set_progress({"files_done": int(self.n), "files_total": int(self.total or 0)})
            return result

    return JobProgress


# Files that hold model weights. allow_model_download: false skips them and still fetches small files.
WEIGHT_PATTERNS = ["*.safetensors", "*.ckpt", "*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "*.h5", "*.msgpack"]


def _targets(entry: ModelEntry) -> list[tuple[str, list[str]]]:
    assert entry.spec is not None
    return [(entry.hf_repo, entry.spec.download_patterns(entry.name)), *entry.spec.extra_downloads()]


def estimate_bytes(entry: ModelEntry) -> int | None:
    """The size of the files a download fetches, from the Hub. None when the Hub does not answer.
    Files that are in the cache already are counted too, so the size is an upper bound."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects
    from mflux.models.common.resolution.model_revisions import ModelRevisions

    total = 0
    try:
        for repo_id, patterns in _targets(entry):
            info = HfApi().model_info(repo_id, revision=ModelRevisions.get(repo_id), files_metadata=True)
            files = filter_repo_objects(info.siblings or [], allow_patterns=patterns, key=lambda f: f.rfilename)
            total += sum(file.size or 0 for file in files)
    except Exception:  # noqa: BLE001 -- no network, or the Hub refuses: the caller asks without a size
        return None
    return total


def download(registry: Registry, entry: ModelEntry, job: Job, with_weights: bool = True) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from mflux.models.common.resolution.model_revisions import ModelRevisions

    if not entry.supported or entry.spec is None:
        raise ApiError("model_not_supported", 422, f"{entry.name}: {entry.reason}", {"model": entry.name})
    targets = _targets(entry)
    progress = _progress_class(job)
    for repo_id, patterns in targets:
        job.set_progress({"phase": "downloading", "repo_id": repo_id})
        snapshot_download(  # ty: ignore[no-matching-overload]
            repo_id=repo_id,
            allow_patterns=patterns,
            ignore_patterns=None if with_weights else WEIGHT_PATTERNS,
            revision=ModelRevisions.get(repo_id),
            tqdm_class=progress,
        )
    return {**entry_for_cache(registry, entry), "weights_skipped": not with_weights}


def delete(registry: Registry, repo_id: str) -> dict[str, Any]:
    from huggingface_hub import scan_cache_dir

    repos = _cache_repos()
    repo = repos.get(repo_id.lower())
    if repo is None:
        raise not_found(f"{repo_id} is not in the Hugging Face cache.", repo_id=repo_id)
    if not registry.by_repo(repo.repo_id):
        raise not_found(f"{repo_id} is not an mflux model repository.", repo_id=repo_id)
    hashes = [revision.commit_hash for revision in repo.revisions]
    strategy = scan_cache_dir().delete_revisions(*hashes)
    freed = strategy.expected_freed_size
    strategy.execute()
    return {"repo_id": repo.repo_id, "freed_gb": round(freed / GB, 2)}
