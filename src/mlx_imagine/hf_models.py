"""The compatible models in the local Hugging Face cache: list, download and delete (docs/api.md 4.3)."""

from datetime import datetime, timezone
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
    for entry in registry.entries().values():
        repo = repos.get(entry.hf_repo.lower())
        if repo is None:
            continue
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


def download(registry: Registry, entry: ModelEntry, job: Job) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from mflux.models.common.resolution.model_revisions import ModelRevisions

    if entry.spec is None:
        raise ApiError("model_not_supported", 422, f"{entry.name}: {entry.reason}", {"model": entry.name})
    targets = [(entry.hf_repo, entry.spec.download_patterns(entry.name))]
    targets += entry.spec.extra_downloads()
    progress = _progress_class(job)
    for repo_id, patterns in targets:
        job.set_progress({"phase": "downloading", "repo_id": repo_id})
        snapshot_download(  # ty: ignore[no-matching-overload]
            repo_id=repo_id,
            allow_patterns=patterns,
            revision=ModelRevisions.get(repo_id),
            tqdm_class=progress,
        )
    return entry_for_cache(registry, entry)


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
