"""The model catalog: the SQLite store, the two scans, the community manifest and the list filters. No network, no
weights: the manifest is a local file and the directories are temporary."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_imagine import catalog as catalog_module, saved_models
from mlx_imagine.api import create_app
from mlx_imagine.catalog import MANIFEST_NAME, Catalog, parse_manifest
from mlx_imagine.engine import Engine

MANIFEST = "\n".join(
    json.dumps(record)
    for record in (
        {
            "stem": "Z-Image-Turbo", "name": "Z-Image Turbo", "model_family": "z-image", "model_type": "image",
            "source": {"repo_id": "Tongyi-MAI/Z-Image-Turbo"}, "release_date": "2025-11-01", "size_gb": 32.9,
            "quants": {
                "q8": {"repo_id": "mflux-community/z-image-turbo-mflux-q8"},
                "q4": {"repo_id": "mflux-community/z-image-turbo-mflux-q4"},
            },
        },
        {"not": "a model record"},
    )
)  # fmt: skip


def _saved(models_dir: Path, name: str, with_sidecar: bool) -> Path:
    path = models_dir / name
    (path / "transformer").mkdir(parents=True)
    (path / "transformer" / saved_models.INDEX).write_text('{"metadata": {"quantization_level": "8"}}')
    if with_sidecar:
        saved_models.write_sidecar(path, "z-image-turbo", "z-image-turbo", 8)
    return path


@pytest.fixture
def catalog(engine: Engine) -> Catalog:
    (engine.catalog.data_dir).mkdir(parents=True, exist_ok=True)
    (engine.catalog.data_dir / MANIFEST_NAME).write_text(MANIFEST)
    return engine.catalog


def test_the_manifest_gives_one_row_for_each_source_and_each_quantization() -> None:
    """Bug: a quantization repo is missing from the manifest rows, so the HF cache scan calls an mflux model a
    plain Hugging Face model."""
    rows = parse_manifest(MANIFEST)
    assert [(row["kind"], row["quant"]) for row in rows] == [("source", None), ("quant", "q8"), ("quant", "q4")]


def test_the_mflux_directory_scan_lists_saved_models_and_flags_a_missing_sidecar(catalog: Catalog) -> None:
    """Bug: a saved model without mlx-imagine.json shows as loadable, and the load then fails with base_model_required."""
    _saved(catalog.settings.path("model_dir"), "with", True)
    _saved(catalog.settings.path("model_dir"), "without", False)
    models = {m["name"]: m for m in catalog.list_models({"store": "mflux_dir"})["models"]}
    assert models["with"]["loadable"] is True and models["with"]["quant"] == "q8"
    assert models["without"]["loadable"] is False and models["without"]["supported"] is False


def test_a_rescan_replaces_the_rows_of_a_store(catalog: Catalog) -> None:
    """Bug: a deleted model stays in the list because a scan only adds rows."""
    path = _saved(catalog.settings.path("model_dir"), "gone", True)
    assert len(catalog.list_models({"store": "mflux_dir"})["models"]) == 1
    import shutil

    shutil.rmtree(path)
    catalog.scan("mflux_dir")
    assert catalog.list_models({"store": "mflux_dir"})["models"] == []


def test_the_remote_view_lists_the_quantizations_of_the_manifest(catalog: Catalog) -> None:
    """Bug: the mflux-community view lists source models, which the user cannot download as mflux models."""
    result = catalog.list_models({"location": "remote", "store": "mflux_community"})
    assert sorted(m["quant"] for m in result["models"]) == ["q4", "q8"]
    assert all(m["format"] == "mflux" and m["location"] == "remote" for m in result["models"])
    assert catalog.list_models({"store": "mflux_community", "format": "hf"})["models"] == []


def test_a_hub_search_needs_a_text_and_hf_access(catalog: Catalog) -> None:
    """Bug: a search of the whole Hub runs with no text, or with allow_hf_access false."""
    from mlx_imagine.errors import ApiError

    with pytest.raises(ApiError) as caught:
        catalog.list_models({"store": "hub"})
    assert caught.value.status == 400
    catalog.settings.hf_allow_access = False
    with pytest.raises(ApiError) as caught:
        catalog.list_models({"store": "hub", "q": "flux"})
    assert caught.value.code == "operation_disabled"


def test_a_location_and_a_store_that_do_not_match_are_refused(catalog: Catalog) -> None:
    """Bug: location=remote with store=mflux_dir returns an empty list, and the client thinks there is no model."""
    from mlx_imagine.errors import ApiError

    with pytest.raises(ApiError) as caught:
        catalog.list_models({"location": "remote", "store": "mflux_dir"})
    assert caught.value.status == 400


def test_the_manifest_sync_does_not_reach_the_network_when_hf_access_is_off(catalog: Catalog, monkeypatch) -> None:
    """Bug: allow_hf_access is false, and the cron thread still downloads the manifest."""
    monkeypatch.setattr(catalog_module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    catalog.settings.hf_allow_access = False
    assert catalog.sync_manifest() == "offline"


def test_the_list_route_filters_and_the_old_routes_are_gone(engine: Engine, catalog: Catalog) -> None:
    """Bug: the list route ignores a filter, or an old route still answers."""
    _saved(engine.models_dir, "m", True)
    client = TestClient(create_app(engine))
    body = client.get("/v1/models/mflux", params={"location": "local", "format": "mflux"}).json()
    assert [m["id"] for m in body["models"]] == ["mflux_dir/m"]
    assert client.get("/v1/models/mflux/mflux_dir/m").json()["name"] == "m"
    assert client.get("/v1/models/mflux/mflux_dir/none").status_code == 404
    assert client.get("/v1/models/mflux", params={"store": "nope"}).status_code == 400
    for old in ("/v1/models/saved", "/v1/models/huggingface", "/v1/models/supported"):
        assert client.get(old).status_code == 404
