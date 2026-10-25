"""The model catalog: what the Hugging Face cache holds (docs/api.md 4).

The cache is the one model store. Each repository in it has one format:

    mflux  an mflux-format model (mflux saved it, often quantized)
    hf     the original Hugging Face layout of a model in the mflux registry
    other  any other repository (a language model, a dataset tool, ...)

A background thread scans the cache every `huggingface_models.cron_minutes` and writes the result to
data/mlx-imagine.sqlite. It also copies the community manifest to data/, which names the base model of each
mflux-community quantization. The list calls read the database. The thread does no MLX work."""

import fnmatch
import json
import logging
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import hf_models, saved_models
from .config import Settings
from .errors import invalid_request, not_found
from .registry import Registry

logger = logging.getLogger(__name__)

DB_NAME = "mlx-imagine.sqlite"
# The mflux-community catalog: models_hf.json lists the repositories on the Hub, models_mflux.json lists the
# mflux models with their upstream repository. The server copies both to data/ (docs/api.md 4.8).
HF_FILE = "models_hf.json"
MFLUX_FILE = "models_mflux.json"
COMMUNITY_REPO = re.compile(r"(?P<owner>[^/]+)/(?P<stem>.+)-mflux-(?P<quant>q\d+|bf16)")
# Short names for an upstream owner in the ns filter.
NS_ALIASES = {"bfl": "black-forest-labs"}
FORMATS = ("mflux", "hf", "other")
# An hf repository below this size holds only small files (configs, a tokenizer), not a model. Only the view
# hf/all lists it.
SMALL_HF_GB = 1.0
# The list views of GET /v8/models/<view>: (formats, True to also list a small hf repository). The default view
# holds the image models.
VIEWS: dict[str, tuple[tuple[str, ...], bool]] = {
    "models": (("mflux", "hf"), False),
    "mflux": (("mflux",), False),
    "hf": (("hf",), False),
    "hf/all": (("hf",), True),
    "other": (("other",), False),
    "all": (FORMATS, False),
}
TICK_SECONDS = 15
MAX_LIMIT = 100

SCHEMA = """
DROP TABLE IF EXISTS models;
CREATE TABLE IF NOT EXISTS cache_models (
    id TEXT PRIMARY KEY, format TEXT NOT NULL, model_name TEXT, base_model TEXT, family TEXT, quant TEXT,
    size_gb REAL, complete INTEGER, supported INTEGER NOT NULL, loadable INTEGER NOT NULL,
    in_manifest INTEGER NOT NULL, path TEXT, extra TEXT, scanned_at TEXT NOT NULL
);
DROP TABLE IF EXISTS manifest;
CREATE TABLE IF NOT EXISTS community (
    repo_id TEXT PRIMARY KEY, kind TEXT NOT NULL, stem TEXT, model_key TEXT, family TEXT, model_type TEXT,
    quant TEXT, source_repo TEXT, ns TEXT, size_gb REAL, upload_date TEXT, commit_hash TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _model_index(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Normalized names -> the key of an mflux model. The repository names of models_hf.json do not always use
    the model key (flux-1-dev is the model dev of the family flux1), so the index has each key, sub-family and
    alias, alone and after the family, and the name of the upstream repository."""
    index: dict[str, str] = {}
    for key, model in models.items():
        family = _key(model.get("model_family") or "")
        for name in (key, model.get("model_sub_family"), *(model.get("model_aliases") or [])):
            if name:
                index.setdefault(_key(name), key)
                index.setdefault(family + _key(name), key)
        upstream = (model.get("upstream") or {}).get("repo")
        if upstream and "/" in upstream:
            index.setdefault(_key(upstream.split("/", 1)[1]), key)
    return index


def parse_community(hf_text: str, mflux_text: str) -> list[dict[str, Any]]:
    """Rows from the two catalog files: one for each upstream repository and one for each repository on the Hub."""
    models: dict[str, dict[str, Any]] = {}
    try:
        for item in json.loads(mflux_text).get("models") or []:
            models.update({key: value for key, value in item.items() if isinstance(value, dict)})
        repos = json.loads(hf_text).get("hf_models") or []
    except (ValueError, AttributeError):
        logger.warning("community: a catalog file is not the expected JSON")
        return []
    rows: dict[str, dict[str, Any]] = {}
    for key, model in models.items():
        upstream = (model.get("upstream") or {}).get("repo")
        if upstream:
            rows[upstream.lower()] = {
                "repo_id": upstream, "kind": "source", "stem": key, "model_key": key,
                "family": model.get("model_family"), "model_type": model.get("model_type"), "quant": None,
                "source_repo": upstream, "ns": upstream.split("/")[0], "size_gb": None, "upload_date": None,
                "commit_hash": None,
            }  # fmt: skip
    index = _model_index(models)
    for repo in repos:
        repo_id = str(repo.get("model_name") or "") if isinstance(repo, dict) else ""
        match = COMMUNITY_REPO.fullmatch(repo_id)
        if match is None:
            logger.warning("community: %r is not an mflux model repository", repo_id)
            continue
        key = index.get(_key(match["stem"]))
        model = models.get(key or "", {})
        upstream = (model.get("upstream") or {}).get("repo")
        rows[repo_id.lower()] = {
            "repo_id": repo_id, "kind": "quant", "stem": match["stem"], "model_key": key,
            "family": model.get("model_family"), "model_type": model.get("model_type"), "quant": match["quant"],
            "source_repo": upstream, "ns": upstream.split("/")[0] if upstream else None,
            "size_gb": repo.get("size_gb"), "upload_date": repo.get("upload_date"),
            "commit_hash": repo.get("commit_hash"),
        }  # fmt: skip
    return list(rows.values())


class Catalog:
    def __init__(self, settings: Settings, registry: Registry) -> None:
        self.settings = settings
        self.registry = registry
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._due = 0.0

    # -- database ----------------------------------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        return self.settings.path("data_dir")

    def _connect(self) -> sqlite3.Connection:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.data_dir / DB_NAME, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
        return connection

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits when the block ends and always closes (sqlite3's own `with` does not close)."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _meta(self, connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # -- community catalog -------------------------------------------------------------------------------------

    def _community_files(self) -> dict[str, str]:
        return {HF_FILE: self.settings.mflux_models_hf_url, MFLUX_FILE: self.settings.mflux_models_mflux_url}

    def load_manifest(self) -> int:
        """Read the two catalog files in data/ into the database. Returns the number of rows."""
        paths = [self.data_dir / name for name in (HF_FILE, MFLUX_FILE)]
        if not all(path.is_file() for path in paths):
            return 0
        rows = parse_community(*(path.read_text(encoding="utf-8") for path in paths))
        with self._lock, self._db() as connection:
            connection.execute("DELETE FROM community")
            connection.executemany(
                "INSERT OR REPLACE INTO community VALUES (:repo_id, :kind, :stem, :model_key, :family, "
                ":model_type, :quant, :source_repo, :ns, :size_gb, :upload_date, :commit_hash)",
                rows,
            )
        return len(rows)

    def sync_manifest(self) -> str:
        """Copy the two catalog files from the Hub to data/ when they changed. Returns 'updated', 'unchanged',
        'offline' or 'failed'. Reaches the network only when huggingface_models.allow_hf_access is true."""
        if not self.settings.hf_allow_access:
            return "offline"
        results = [self._sync_file(name, url) for name, url in self._community_files().items()]
        if "updated" in results:
            self.load_manifest()
            with self._lock, self._db() as connection:
                self._set_meta(connection, "community_synced_at", _now())
            return "updated"
        return "failed" if "failed" in results else "unchanged"

    def _sync_file(self, name: str, url: str) -> str:
        path = self.data_dir / name
        with self._lock, self._db() as connection:
            etag = self._meta(connection, f"etag_{name}") if path.is_file() else None
        request = urllib.request.Request(url, headers={"If-None-Match": etag} if etag else {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                new_etag = response.headers.get("ETag")
        except urllib.error.HTTPError as error:
            if error.code == 304:
                return "unchanged"
            logger.warning("community: the download of %s failed with HTTP %s", name, error.code)
            return "failed"
        except (urllib.error.URLError, OSError) as error:
            logger.warning("community: the download of %s failed: %s", name, error)
            return "failed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(body)
        partial.replace(path)
        with self._lock, self._db() as connection:
            self._set_meta(connection, f"etag_{name}", new_etag or "")
        return "updated"

    def _manifest_index(self, connection: sqlite3.Connection) -> dict[str, sqlite3.Row]:
        return {row["repo_id"].lower(): row for row in connection.execute("SELECT * FROM community")}

    def list_community(self, filters: dict[str, Any]) -> dict[str, Any]:
        """The mflux-community repositories on the Hub that match the filters: quant, ns, search, limit, cursor."""
        cursor = str(filters.get("cursor") or "0")
        if not cursor.isdigit():
            raise invalid_request("cursor must be a value from next_cursor.", cursor=cursor)
        offset = int(cursor)
        try:
            limit = min(max(int(filters.get("limit") or MAX_LIMIT), 1), MAX_LIMIT)
        except (TypeError, ValueError):
            raise invalid_request("limit must be a whole number.", limit=filters.get("limit")) from None
        with self._lock, self._db() as connection:
            empty = connection.execute("SELECT 1 FROM community LIMIT 1").fetchone() is None
        if empty:
            self.load_manifest()
        clauses, params = ["kind = 'quant'"], []
        if filters.get("quant"):
            quant = str(filters["quant"]).lower()
            clauses.append("quant = ?")
            params.append(quant if quant == "bf16" or quant.startswith("q") else f"q{quant}")
        if filters.get("ns"):
            ns = str(filters["ns"]).lower()
            clauses.append("lower(ns) = ?")
            params.append(NS_ALIASES.get(ns, ns))
        sql = f"SELECT * FROM community WHERE {' AND '.join(clauses)} ORDER BY lower(repo_id)"  # noqa: S608
        with self._lock, self._db() as connection:
            rows = list(connection.execute(sql, params))
            cached = {row["id"].lower() for row in connection.execute("SELECT id FROM cache_models")}
        search = str(filters.get("search") or "").lower()
        if search:
            # A text anywhere in the ID. A * inside the text matches any text: qwen*2-1 finds qwen-image-2-1.
            rows = [row for row in rows if fnmatch.fnmatchcase(row["repo_id"].lower(), f"*{search}*")]
        items = [self._community_item(row, row["repo_id"].lower() in cached) for row in rows]
        page = items[offset : offset + limit]
        more = offset + limit < len(items)
        return {"models": page, "total": len(items), "next_cursor": str(offset + limit) if more else None}

    def _community_item(self, row: sqlite3.Row, in_cache: bool) -> dict[str, Any]:
        entry = self._supported_entry(row["source_repo"])
        return {
            "id": row["repo_id"], "repo_id": row["repo_id"], "quant": row["quant"], "size_gb": row["size_gb"],
            "model_key": row["model_key"], "model_name": entry.name if entry else None, "family": row["family"],
            "model_type": row["model_type"], "upstream_repo": row["source_repo"], "ns": row["ns"],
            "supported": bool(entry and entry.supported), "in_cache": in_cache, "upload_date": row["upload_date"],
            "commit_hash": row["commit_hash"], "url": f"https://huggingface.co/{row['repo_id']}",
        }  # fmt: skip

    # -- scan --------------------------------------------------------------------------------------------------

    def scan(self) -> int:
        """Rescan the Hugging Face cache and replace all rows. Returns the number of repositories."""
        with self._lock, self._db() as connection:
            manifest = self._manifest_index(connection)
            rows = [self._scan_repo(repo, manifest) for repo in hf_models._cache_repos().values()]
            connection.execute("DELETE FROM cache_models")
            connection.executemany(
                "INSERT OR REPLACE INTO cache_models VALUES (:id, :format, :model_name, :base_model, :family, "
                ":quant, :size_gb, :complete, :supported, :loadable, :in_manifest, :path, :extra, :scanned_at)",
                rows,
            )
            self._set_meta(connection, "scanned_cache_models", _now())
        return len(rows)

    def _row(self, repo_id: str, **fields: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": repo_id, "format": "other", "model_name": None, "base_model": None, "family": None,
            "quant": None, "size_gb": None, "complete": None, "supported": 0, "loadable": 0, "in_manifest": 0,
            "path": None, "extra": None, "scanned_at": _now(),
        }  # fmt: skip
        base.update(fields)
        for key in ("complete", "supported", "loadable", "in_manifest"):
            base[key] = None if base[key] is None else int(bool(base[key]))
        return base

    def _supported_entry(self, repo_id: str | None) -> Any:
        entries = self.registry.by_repo(repo_id) if repo_id else []
        return next((e for e in entries if e.supported), entries[0] if entries else None)

    def _scan_repo(self, repo: Any, manifest: dict[str, sqlite3.Row]) -> dict[str, Any]:
        known = manifest.get(repo.repo_id.lower())
        is_quant = known is not None and known["kind"] == "quant"
        snapshot = hf_models.latest_snapshot(repo)
        extra = {
            "revisions": sorted(revision.commit_hash[:7] for revision in repo.revisions),
            "last_accessed": hf_models._iso(repo.last_accessed),
        }
        fields: dict[str, Any] = {
            "size_gb": round(repo.size_on_disk / 1000**3, 2),
            "in_manifest": known is not None,
            "path": str(snapshot) if snapshot else None,
        }
        registry_entry = self._supported_entry(repo.repo_id)
        if registry_entry is None and snapshot is not None and saved_models.is_mflux_layout(snapshot):
            info = saved_models.describe(snapshot)
            # The sidecar names the base model. For a community quantization, the manifest names it.
            base = info["base_model"] if info["base_model"] and self._known(info["base_model"]) else None
            entry = self.registry.find(base) if base else None
            if entry is None and is_quant:
                entry = self._supported_entry(known["source_repo"])
            quant = known["quant"] if is_quant else None
            extra.update(
                {
                    "mflux_version": info["mflux_version"],
                    "saved_at": info["saved_at"],
                    "has_sidecar": info["has_sidecar"],
                }
            )
            fields.update(
                format="mflux",
                model_name=entry.name if entry else None,
                base_model=entry.name if entry else None,
                quant=quant or (f"q{info['quantize']}" if info["quantize"] else None),
                complete=True,
                supported=bool(entry and entry.supported),
                loadable=bool(entry and entry.supported),
            )
        elif registry_entry is not None:
            entry = registry_entry
            complete = hf_models.is_complete(entry)
            fields.update(
                format="hf",
                model_name=entry.name,
                complete=complete,
                supported=entry.supported,
                loadable=bool(entry.supported and complete),
            )
        else:
            entry = None
        family = known["family"] if known else None
        fields["family"] = family or (entry.spec.family if entry and entry.spec else None)
        return self._row(repo.repo_id, extra=json.dumps(extra), **fields)

    def quant_info(self, repo_id: str) -> sqlite3.Row | None:
        """The community row of a quantization repository, or None."""
        with self._lock, self._db() as connection:
            return connection.execute(
                "SELECT * FROM community WHERE kind = 'quant' AND lower(repo_id) = ?", (repo_id.lower(),)
            ).fetchone()

    def quant_source(self, repo_id: str) -> str | None:
        """The upstream repository of a community quantization repository, or None."""
        row = self.quant_info(repo_id)
        return row["source_repo"] if row else None

    def _known(self, name: str) -> bool:
        try:
            self.registry.find(name)
        except Exception:  # noqa: BLE001 -- a stale sidecar names a model that mflux dropped
            return False
        return True

    def scan_all(self) -> None:
        self.sync_manifest()
        self.load_manifest()
        try:
            self.scan()
        except Exception:
            logger.exception("catalog: the scan of the Hugging Face cache failed")

    # -- background thread -------------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None or not self.settings.hf_cron_minutes:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="catalog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _run(self) -> None:
        # The first pass runs at once. The manifest sync comes before each scan.
        self.load_manifest()
        while not self._stop.is_set():
            minutes = self.settings.hf_cron_minutes
            now = time.monotonic()
            if minutes > 0 and self._due <= now:
                self._due = now + minutes * 60
                try:
                    self.sync_manifest()
                    self.scan()
                except Exception:
                    logger.exception("catalog: the scan of the Hugging Face cache failed")
            self._stop.wait(TICK_SECONDS)

    # -- queries -----------------------------------------------------------------------------------------------

    def _ensure_scanned(self) -> None:
        with self._lock, self._db() as connection:
            scanned = self._meta(connection, "scanned_cache_models")
        if scanned is None:
            self.load_manifest()
            self.scan()

    def list_models(self, view: str, filters: dict[str, Any]) -> dict[str, Any]:
        """The models of one view (VIEWS) that match the filters: supported, family, q, limit, cursor."""
        if view not in VIEWS:
            raise not_found(f"{view!r} is not a model list. Use one of {', '.join(VIEWS)}.", view=view)
        cursor = str(filters.get("cursor") or "0")
        if not cursor.isdigit():
            raise invalid_request("cursor must be a value from next_cursor.", cursor=cursor)
        offset = int(cursor)
        try:
            limit = min(max(int(filters.get("limit") or MAX_LIMIT), 1), MAX_LIMIT)
        except (TypeError, ValueError):
            raise invalid_request("limit must be a whole number.", limit=filters.get("limit")) from None
        self._ensure_scanned()
        items = self._items(*VIEWS[view], filters)
        page = items[offset : offset + limit]
        more = offset + limit < len(items)
        return {"models": page, "total": len(items), "next_cursor": str(offset + limit) if more else None}

    def view_items(self, view: str) -> list[dict[str, Any]]:
        """All rows of one view, with no page limit."""
        return self._items(*VIEWS[view], {})

    def _items(self, formats: tuple[str, ...], small_hf: bool, filters: dict[str, Any]) -> list[dict[str, Any]]:
        clauses = [f"format IN ({', '.join('?' for _ in formats)})"]
        params: list[Any] = list(formats)
        if not small_hf:
            clauses.append("NOT (format = 'hf' AND coalesce(size_gb, 0) < ?)")
            params.append(SMALL_HF_GB)
        if filters.get("family") is not None:
            clauses.append("family = ?")
            params.append(filters["family"])
        if filters.get("supported") is not None:
            clauses.append("supported = ?")
            params.append(int(bool(filters["supported"])))
        if filters.get("q"):
            clauses.append("lower(id) LIKE ?")
            params.append(f"%{str(filters['q']).lower()}%")
        sql = f"SELECT * FROM cache_models WHERE {' AND '.join(clauses)} ORDER BY format, lower(id)"  # noqa: S608
        with self._lock, self._db() as connection:
            return [self._item(row) for row in connection.execute(sql, params)]

    def _item(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        extra = json.loads(item.pop("extra") or "{}")
        for key in ("complete", "supported", "loadable", "in_manifest"):
            item[key] = None if item[key] is None else bool(item[key])
        return {"id": item["id"], "repo_id": item["id"], **item, **extra}

    def get(self, repo_id: str) -> dict[str, Any]:
        """One repository. A repository that the last scan did not see gets one new scan, so a model that
        arrived since then is found at once."""
        self._ensure_scanned()
        row = self._get_row(repo_id)
        if row is None:
            self.scan()
            row = self._get_row(repo_id)
        if row is None:
            raise not_found(f"{repo_id} is not in the Hugging Face cache.", repo_id=repo_id)
        return self._item(row)

    def _get_row(self, repo_id: str) -> sqlite3.Row | None:
        with self._lock, self._db() as connection:
            return connection.execute("SELECT * FROM cache_models WHERE lower(id) = ?", (repo_id.lower(),)).fetchone()

    def status(self) -> dict[str, Any]:
        with self._lock, self._db() as connection:
            return {
                "database": str(self.data_dir / DB_NAME),
                "hf_cache_scanned_at": self._meta(connection, "scanned_cache_models"),
                "community_synced_at": self._meta(connection, "community_synced_at"),
            }
