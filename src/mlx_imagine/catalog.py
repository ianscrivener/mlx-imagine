"""The model catalog: what the two local directories and the community manifest hold (docs/api.md 4).

A background thread scans the Hugging Face cache and the mflux model directory every `cron_minutes` and writes the
result to data/mlx-imagine.sqlite. It also copies the community manifest to data/. The list calls read the
database. The thread does no MLX work."""

import json
import logging
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
from .errors import ApiError, invalid_request, not_found, unprocessable
from .registry import Registry

logger = logging.getLogger(__name__)

DB_NAME = "mlx-imagine.sqlite"
MANIFEST_NAME = "models_community_manifest.jsonl"
STORES = ("hf_cache", "mflux_dir", "mflux_community", "hub")
LOCAL_STORES = ("hf_cache", "mflux_dir")
REMOTE_STORES = ("mflux_community", "hub")
FORMATS = ("hf", "mflux")
TICK_SECONDS = 15
MAX_LIMIT = 100

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY, store TEXT NOT NULL, location TEXT NOT NULL, format TEXT NOT NULL,
    repo_id TEXT, name TEXT NOT NULL, model_name TEXT, base_model TEXT, family TEXT, quant TEXT,
    size_gb REAL, complete INTEGER, supported INTEGER NOT NULL, loadable INTEGER NOT NULL,
    in_manifest INTEGER NOT NULL, path TEXT, extra TEXT, scanned_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manifest (
    repo_id TEXT PRIMARY KEY, kind TEXT NOT NULL, stem TEXT, name TEXT, family TEXT, model_type TEXT,
    quant TEXT, source_repo TEXT, size_gb REAL, release_date TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_manifest(text: str) -> list[dict[str, Any]]:
    """Manifest rows from the JSON Lines text: one row for each source model and one for each quantization."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            source = item["source"]["repo_id"]
        except (ValueError, KeyError, TypeError):
            logger.warning("manifest: a line was skipped because it is not a model record")
            continue
        common = {
            "stem": item.get("stem"),
            "name": item.get("name"),
            "family": item.get("model_family"),
            "model_type": item.get("model_type"),
            "source_repo": source,
            "release_date": item.get("release_date"),
        }
        rows.append({**common, "repo_id": source, "kind": "source", "quant": None, "size_gb": item.get("size_gb")})
        for quant, info in (item.get("quants") or {}).items():
            if isinstance(info, dict) and info.get("repo_id"):
                rows.append({**common, "repo_id": info["repo_id"], "kind": "quant", "quant": quant, "size_gb": None})
    return rows


class Catalog:
    def __init__(self, settings: Settings, registry: Registry) -> None:
        self.settings = settings
        self.registry = registry
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._due: dict[str, float] = {}

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

    # -- manifest ----------------------------------------------------------------------------------------------

    def load_manifest(self) -> int:
        """Read data/<manifest> into the database. Returns the number of rows."""
        path = self.data_dir / MANIFEST_NAME
        if not path.is_file():
            return 0
        rows = parse_manifest(path.read_text(encoding="utf-8"))
        with self._lock, self._db() as connection:
            connection.execute("DELETE FROM manifest")
            connection.executemany(
                "INSERT OR REPLACE INTO manifest VALUES (:repo_id, :kind, :stem, :name, :family, :model_type, "
                ":quant, :source_repo, :size_gb, :release_date)",
                rows,
            )
        return len(rows)

    def sync_manifest(self) -> str:
        """Copy the manifest from the Hub to data/ when it changed. Returns 'updated', 'unchanged', 'offline'
        or 'failed'. Reaches the network only when huggingface_models.allow_hf_access is true."""
        if not self.settings.hf_allow_access:
            return "offline"
        path = self.data_dir / MANIFEST_NAME
        with self._lock, self._db() as connection:
            etag = self._meta(connection, "manifest_etag") if path.is_file() else None
        headers = {"If-None-Match": etag} if etag else {}
        request = urllib.request.Request(self.settings.mflux_manifest_url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
                new_etag = response.headers.get("ETag")
        except urllib.error.HTTPError as error:
            if error.code == 304:
                return "unchanged"
            logger.warning("manifest: the download failed with HTTP %s", error.code)
            return "failed"
        except (urllib.error.URLError, OSError) as error:
            logger.warning("manifest: the download failed: %s", error)
            return "failed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(body)
        partial.replace(path)
        self.load_manifest()
        with self._lock, self._db() as connection:
            self._set_meta(connection, "manifest_etag", new_etag or "")
            self._set_meta(connection, "manifest_synced_at", _now())
        return "updated"

    def _manifest_index(self, connection: sqlite3.Connection) -> dict[str, sqlite3.Row]:
        return {row["repo_id"].lower(): row for row in connection.execute("SELECT * FROM manifest")}

    # -- scans -------------------------------------------------------------------------------------------------

    def scan(self, store: str) -> int:
        """Rescan one local store and replace its rows. Returns the number of models."""
        if store not in LOCAL_STORES:
            raise invalid_request(f"{store!r} is not a local store.", store=store)
        with self._lock, self._db() as connection:
            manifest = self._manifest_index(connection)
            rows = self._scan_hf_cache(manifest) if store == "hf_cache" else self._scan_mflux_dir()
            connection.execute("DELETE FROM models WHERE store = ?", (store,))
            connection.executemany(
                "INSERT OR REPLACE INTO models VALUES (:id, :store, :location, :format, :repo_id, :name, "
                ":model_name, :base_model, :family, :quant, :size_gb, :complete, :supported, :loadable, "
                ":in_manifest, :path, :extra, :scanned_at)",
                rows,
            )
            self._set_meta(connection, f"scanned_{store}", _now())
        return len(rows)

    def _row(self, store: str, name: str, **fields: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": f"{store}/{name}", "store": store, "location": "local", "format": "hf", "repo_id": None,
            "name": name, "model_name": None, "base_model": None, "family": None, "quant": None, "size_gb": None,
            "complete": None, "supported": 0, "loadable": 0, "in_manifest": 0, "path": None, "extra": None,
            "scanned_at": _now(),
        }  # fmt: skip
        base.update(fields)
        for key in ("complete", "supported", "loadable", "in_manifest"):
            base[key] = None if base[key] is None else int(bool(base[key]))
        return base

    def _scan_hf_cache(self, manifest: dict[str, sqlite3.Row]) -> list[dict[str, Any]]:
        rows = []
        for key, repo in hf_models._cache_repos().items():
            known = manifest.get(key)
            entries = self.registry.by_repo(repo.repo_id)
            entry = next((e for e in entries if e.supported), entries[0] if entries else None)
            if known is None and entry is None:
                continue  # not an mflux model (a later, wider scan can list it)
            is_quant = known is not None and known["kind"] == "quant"
            if is_quant:
                source = self.registry.by_repo(known["source_repo"])
                entry = next((e for e in source if e.supported), source[0] if source else None)
            snapshot = hf_models.latest_snapshot(repo)
            if is_quant:
                complete = snapshot is not None and saved_models.is_saved_model(snapshot)
            else:
                complete = None if entry is None else hf_models.is_complete(entry)
            extra = {
                "revisions": sorted(revision.commit_hash[:7] for revision in repo.revisions),
                "last_accessed": hf_models._iso(repo.last_accessed),
            }
            rows.append(
                self._row(
                    "hf_cache", repo.repo_id, repo_id=repo.repo_id, format="mflux" if is_quant else "hf",
                    model_name=entry.name if entry else None, family=known["family"] if known else None,
                    quant=known["quant"] if is_quant else None, size_gb=round(repo.size_on_disk / 1000**3, 2),
                    complete=complete, supported=bool(entry and entry.supported),
                    loadable=bool(entry and entry.supported and complete),
                    in_manifest=known is not None, extra=json.dumps(extra),
                )
            )  # fmt: skip
        return rows

    def _scan_mflux_dir(self) -> list[dict[str, Any]]:
        rows = []
        for item in saved_models.list_models(self.settings.path("model_dir")):
            base = item["base_model"]
            entry = self.registry.find(base) if base and self._known(base) else None
            quant = f"q{item['quantize']}" if item["quantize"] else None
            rows.append(
                self._row(
                    "mflux_dir", item["name"], format="mflux", model_name=item["model_name"], base_model=base,
                    family=entry.spec.family if entry and entry.spec else None, quant=quant,
                    size_gb=item["size_gb"], supported=bool(entry and entry.supported),
                    loadable=bool(entry and entry.supported and item["has_sidecar"]), path=item["path"],
                    extra=json.dumps({"mflux_version": item["mflux_version"], "saved_at": item["saved_at"],
                                      "has_sidecar": item["has_sidecar"]}),
                )
            )  # fmt: skip
        return rows

    def quant_source(self, repo_id: str) -> str | None:
        """The source repository that the manifest gives for a quantization repository, or None."""
        with self._lock, self._db() as connection:
            row = connection.execute(
                "SELECT source_repo FROM manifest WHERE kind = 'quant' AND lower(repo_id) = ?", (repo_id.lower(),)
            ).fetchone()
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
        for store in LOCAL_STORES:
            try:
                self.scan(store)
            except Exception:
                logger.exception("catalog: the scan of %s failed", store)

    # -- background thread -------------------------------------------------------------------------------------

    def _interval(self, store: str) -> int:
        return self.settings.hf_cron_minutes if store == "hf_cache" else self.settings.mflux_cron_minutes

    def start(self) -> None:
        if self._thread is not None or not (self.settings.mflux_cron_minutes or self.settings.hf_cron_minutes):
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
        # The first pass runs at once. Each store has its own interval; the manifest follows the cache scan.
        self.load_manifest()
        while not self._stop.is_set():
            now = time.monotonic()
            for store in LOCAL_STORES:
                minutes = self._interval(store)
                if minutes <= 0 or self._due.get(store, 0) > now:
                    continue
                self._due[store] = now + minutes * 60
                try:
                    if store == "hf_cache":
                        self.sync_manifest()
                    self.scan(store)
                except Exception:
                    logger.exception("catalog: the scan of %s failed", store)
            self._stop.wait(TICK_SECONDS)

    # -- queries -----------------------------------------------------------------------------------------------

    def _ensure_scanned(self, store: str) -> None:
        with self._lock, self._db() as connection:
            scanned = self._meta(connection, f"scanned_{store}")
        if scanned is None:
            self.load_manifest()
            self.scan(store)

    def list_models(self, filters: dict[str, Any]) -> dict[str, Any]:
        """The models that match the filters: location, format, store, supported, family, q, limit, cursor."""
        store = filters.get("store")
        location = filters.get("location")
        if store is not None and store not in STORES:
            raise invalid_request(f"store must be one of {', '.join(STORES)}.", store=store)
        if location is not None and location not in ("local", "remote"):
            raise invalid_request("location must be 'local' or 'remote'.", location=location)
        if filters.get("format") is not None and filters["format"] not in FORMATS:
            raise invalid_request("format must be 'hf' or 'mflux'.", format=filters["format"])
        remote = location == "remote" or store in REMOTE_STORES
        if remote and (location == "local" or store in LOCAL_STORES):
            raise invalid_request("location and store do not match.", location=location, store=store)
        cursor = str(filters.get("cursor") or "0")
        if not cursor.isdigit():
            raise invalid_request("cursor must be a value from next_cursor.", cursor=cursor)
        offset = int(cursor)
        try:
            limit = min(max(int(filters.get("limit") or (20 if remote else MAX_LIMIT)), 1), MAX_LIMIT)
        except (TypeError, ValueError):
            raise invalid_request("limit must be a whole number.", limit=filters.get("limit")) from None
        if remote and store == "hub":
            return self._search_hub(filters, limit, offset)
        if remote:
            items = self._manifest_items(filters)
        else:
            for name in LOCAL_STORES:
                if store in (None, name):
                    self._ensure_scanned(name)
            items = self._local_items(filters)
        page = items[offset : offset + limit]
        more = offset + limit < len(items)
        return {"models": page, "total": len(items), "next_cursor": str(offset + limit) if more else None}

    def _local_items(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        clauses, params = ["location = 'local'"], []
        for column in ("store", "format", "family"):
            if filters.get(column) is not None:
                clauses.append(f"{column} = ?")
                params.append(filters[column])
        if filters.get("supported") is not None:
            clauses.append("supported = ?")
            params.append(int(bool(filters["supported"])))
        if filters.get("q"):
            clauses.append("(lower(name) LIKE ? OR lower(coalesce(repo_id, '')) LIKE ?)")
            params += [f"%{str(filters['q']).lower()}%"] * 2
        sql = f"SELECT * FROM models WHERE {' AND '.join(clauses)} ORDER BY store, name"  # noqa: S608
        with self._lock, self._db() as connection:
            return [self._item(row) for row in connection.execute(sql, params)]

    def _item(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        extra = json.loads(item.pop("extra") or "{}")
        for key in ("complete", "supported", "loadable", "in_manifest"):
            item[key] = None if item[key] is None else bool(item[key])
        return {**item, **extra}

    def _manifest_items(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._has_manifest():
            self.load_manifest()
        if filters.get("format") == "hf":
            return []
        sql = "SELECT * FROM manifest WHERE kind = 'quant' ORDER BY release_date DESC, repo_id"
        with self._lock, self._db() as connection:
            rows = list(connection.execute(sql))
        items = []
        for row in rows:
            if filters.get("family") and row["family"] != filters["family"]:
                continue
            if filters.get("q") and str(filters["q"]).lower() not in row["repo_id"].lower():
                continue
            entries = self.registry.by_repo(row["source_repo"])
            entry = next((e for e in entries if e.supported), entries[0] if entries else None)
            supported = bool(entry and entry.supported)
            if filters.get("supported") is not None and supported != bool(filters["supported"]):
                continue
            items.append(
                {
                    "id": f"mflux_community/{row['repo_id']}", "store": "mflux_community", "location": "remote",
                    "format": "mflux", "repo_id": row["repo_id"], "name": row["name"], "model_name": entry.name if entry else None,
                    "family": row["family"], "quant": row["quant"], "supported": supported, "loadable": False,
                    "in_manifest": True, "source_repo": row["source_repo"], "release_date": row["release_date"],
                    "url": f"https://huggingface.co/{row['repo_id']}",
                }
            )  # fmt: skip
        return items

    def _has_manifest(self) -> bool:
        with self._lock, self._db() as connection:
            return connection.execute("SELECT 1 FROM manifest LIMIT 1").fetchone() is not None

    def _search_hub(self, filters: dict[str, Any], limit: int, offset: int) -> dict[str, Any]:
        """A live search of the whole Hub (view 4). It needs a text and huggingface_models.allow_hf_access."""
        query = filters.get("q")
        if not query:
            raise invalid_request("A search of the Hub needs 'q'.")
        if not self.settings.hf_allow_access:
            raise ApiError("operation_disabled", 403, "huggingface_models.allow_hf_access is false.")
        from itertools import islice

        from huggingface_hub import HfApi

        try:
            found = list(
                islice(HfApi().list_models(search=query, sort="downloads", limit=offset + limit + 1), offset, None)
            )
        except Exception as error:  # noqa: BLE001 -- the Hub or the network failed
            raise unprocessable("hub_unavailable", f"The Hub search failed: {error}") from None
        with self._lock, self._db() as connection:
            manifest = self._manifest_index(connection)
        items = []
        for model in found[:limit]:
            known = manifest.get(model.id.lower())
            is_quant = known is not None and known["kind"] == "quant"
            entries = self.registry.by_repo(model.id)
            items.append(
                {
                    "supported": any(e.supported for e in entries), "loadable": False,
                    "id": f"hub/{model.id}", "store": "hub", "location": "remote",
                    "format": "mflux" if is_quant else "hf", "repo_id": model.id, "name": model.id,
                    "family": known["family"] if known else None, "quant": known["quant"] if is_quant else None,
                    "in_manifest": known is not None, "downloads": model.downloads, "url": f"https://huggingface.co/{model.id}",
                }
            )  # fmt: skip
        return {"models": items, "total": None, "next_cursor": str(offset + limit) if len(found) > limit else None}

    def get(self, store: str, name: str) -> dict[str, Any]:
        if store not in STORES:
            raise not_found(f"{store!r} is not a store.", store=store)
        if store in REMOTE_STORES:
            result = self.list_models({"store": store, "q": name, "limit": MAX_LIMIT})
            for item in result["models"]:
                if item["repo_id"] == name:
                    return item
        else:
            self._ensure_scanned(store)
            with self._lock, self._db() as connection:
                row = connection.execute("SELECT * FROM models WHERE id = ?", (f"{store}/{name}",)).fetchone()
            if row is not None:
                return self._item(row)
        raise not_found(f"No model {name!r} in the store {store!r}.", store=store, name=name)

    def status(self) -> dict[str, Any]:
        with self._lock, self._db() as connection:
            return {
                "database": str(self.data_dir / DB_NAME),
                "hf_cache_scanned_at": self._meta(connection, "scanned_hf_cache"),
                "mflux_dir_scanned_at": self._meta(connection, "scanned_mflux_dir"),
                "manifest_synced_at": self._meta(connection, "manifest_synced_at"),
            }
