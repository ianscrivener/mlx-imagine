"""The model catalog: the SQLite store, the scan of the Hugging Face cache, the community manifest and the list
views. No network, no weights: the manifest is a local file and the cache is temporary."""

import json
from pathlib import Path

import pytest
from conftest import cache_repo, wait_for
from fastapi.testclient import TestClient

from mlx_imagine import catalog as catalog_module, hf_models, saved_models
from mlx_imagine.api import create_app
from mlx_imagine.catalog import HF_FILE, MFLUX_FILE, Catalog, parse_community
from mlx_imagine.engine import Engine

MODELS_MFLUX = json.dumps(
    {
        "metadata": {"datetime_extract": "2026-10-19T05:17:12+00:00"},
        "models": [
            {"z-image-turbo": {"model_family": "z-image", "model_type": "image", "model_aliases": ["zit"],
                               "upstream": {"repo": "Tongyi-MAI/Z-Image-Turbo"}, "quants": ["q4", "q8"]}},
            {"dev": {"model_family": "flux1", "model_type": "image", "model_aliases": [],
                     "upstream": {"repo": "black-forest-labs/FLUX.1-dev"}, "quants": ["q4"]}},
        ],
    }
)  # fmt: skip
MODELS_HF = json.dumps(
    {
        "hf_models": [
            {"model_name": "mflux-community/z-image-turbo-mflux-q8", "size_gb": 9.1, "upload_date": "2026-08-01",
             "upload_user": "mflux-community", "commit_hash": "a" * 40},
            {"model_name": "mflux-community/z-image-turbo-mflux-q4", "size_gb": 5.2, "upload_date": "2026-08-01",
             "upload_user": "mflux-community", "commit_hash": "b" * 40},
            {"model_name": "mflux-community/flux-1-dev-mflux-q4", "size_gb": 6.9, "upload_date": "2026-08-02",
             "upload_user": "mflux-community", "commit_hash": "c" * 40},
            {"model_name": "mflux-community/new-model-mflux-bf16", "size_gb": 20.0, "upload_date": "2026-08-03",
             "upload_user": "mflux-community", "commit_hash": "d" * 40},
            {"model_name": "not a repository"},
        ]
    }
)  # fmt: skip


@pytest.fixture
def catalog(engine: Engine) -> Catalog:
    (engine.catalog.data_dir).mkdir(parents=True, exist_ok=True)
    (engine.catalog.data_dir / HF_FILE).write_text(MODELS_HF)
    (engine.catalog.data_dir / MFLUX_FILE).write_text(MODELS_MFLUX)
    return engine.catalog


def test_the_community_files_give_one_row_for_each_source_and_each_repository() -> None:
    """Bug: a repository of models_hf.json does not find its mflux model (flux-1-dev is the model dev), so the
    server does not know its upstream repository or its base model."""
    rows = {row["repo_id"]: row for row in parse_community(MODELS_HF, MODELS_MFLUX)}
    assert rows["Tongyi-MAI/Z-Image-Turbo"]["kind"] == "source"
    assert rows["mflux-community/flux-1-dev-mflux-q4"]["source_repo"] == "black-forest-labs/FLUX.1-dev"
    assert rows["mflux-community/flux-1-dev-mflux-q4"]["ns"] == "black-forest-labs"
    assert rows["mflux-community/new-model-mflux-bf16"]["source_repo"] is None  # no mflux model yet
    assert rows["mflux-community/new-model-mflux-bf16"]["quant"] == "bf16"
    assert "not a repository" not in rows


def test_the_community_list_filters_by_quant_ns_and_search(engine: Engine, catalog: Catalog) -> None:
    """Bug: a filter of /v8/models-community does nothing, or q=4 does not mean the quantization q4."""
    client = TestClient(create_app(engine))

    def ids(**params: str) -> list[str]:
        return [m["id"] for m in client.get("/v8/models-community", params=params).json()["models"]]

    assert len(ids()) == 4
    assert ids(q="4") == ["mflux-community/flux-1-dev-mflux-q4", "mflux-community/z-image-turbo-mflux-q4"]
    assert ids(quant="q8") == ["mflux-community/z-image-turbo-mflux-q8"]
    assert ids(quant="bf16") == ["mflux-community/new-model-mflux-bf16"]
    assert ids(ns="BFL") == ["mflux-community/flux-1-dev-mflux-q4"]
    assert ids(namespace="tongyi-mai", q="8") == ["mflux-community/z-image-turbo-mflux-q8"]
    assert ids(search="z-image") == ids(s="Z-IMAGE") == ids(ns="Tongyi-MAI")
    assert ids(search="turbo-mflux-q8") == ["mflux-community/z-image-turbo-mflux-q8"]
    assert ids(s="dev*q4") == ["mflux-community/flux-1-dev-mflux-q4"]
    item = client.get("/v8/models-community", params={"quant": "8"}).json()["models"][0]
    assert item["model_name"] == "z-image-turbo" and item["supported"] is True and item["in_cache"] is False


def test_the_sync_reads_both_files_and_sends_the_etag(catalog: Catalog, monkeypatch) -> None:
    """Bug: the cron sync fetches only one file, or fetches both again each time, with no If-None-Match."""
    import io
    import urllib.error

    for name in (HF_FILE, MFLUX_FILE):
        (catalog.data_dir / name).unlink()
    seen: list[tuple[str, str | None]] = []

    class Response(io.BytesIO):
        def __init__(self, body: str, etag: str) -> None:
            super().__init__(body.encode())
            self.headers = {"ETag": etag}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen.append((request.full_url.rsplit("/", 1)[-1], request.headers.get("If-none-match")))
        if request.headers.get("If-none-match"):
            raise urllib.error.HTTPError(request.full_url, 304, "Not Modified", {}, None)
        return Response(MODELS_HF if request.full_url.endswith(HF_FILE) else MODELS_MFLUX, f"e-{len(seen)}")

    monkeypatch.setattr(catalog_module.urllib.request, "urlopen", urlopen)
    assert catalog.sync_manifest() == "updated"
    assert catalog.quant_source("mflux-community/z-image-turbo-mflux-q8") == "Tongyi-MAI/Z-Image-Turbo"
    assert catalog.sync_manifest() == "unchanged"
    assert seen == [(HF_FILE, None), (MFLUX_FILE, None), (HF_FILE, "e-1"), (MFLUX_FILE, "e-2")]


def test_the_scan_sorts_each_repository_into_one_format(catalog: Catalog) -> None:
    """Bug: an mflux-format model shows as an hf model, an unknown repository shows as an image model, or a
    model with no base model shows as loadable, and the load then fails with base_model_required."""
    cache_repo("local/with", with_sidecar=True)
    cache_repo("local/without")
    cache_repo("mflux-community/z-image-turbo-mflux-q8")
    cache_repo("Tongyi-MAI/Z-Image-Turbo", mflux=False, size_gb=2)
    cache_repo("someone/a-language-model", mflux=False)
    catalog.load_manifest()
    models = {m["id"]: m for m in catalog.list_models("all", {})["models"]}
    assert {key: m["format"] for key, m in models.items()} == {
        "local/with": "mflux", "local/without": "mflux", "mflux-community/z-image-turbo-mflux-q8": "mflux",
        "Tongyi-MAI/Z-Image-Turbo": "hf", "someone/a-language-model": "other",
    }  # fmt: skip
    assert models["local/with"]["loadable"] is True and models["local/with"]["quant"] == "q8"
    assert models["local/without"]["loadable"] is False and models["local/without"]["supported"] is False
    community = models["mflux-community/z-image-turbo-mflux-q8"]
    assert community["base_model"] == "z-image-turbo" and community["loadable"] is True
    assert models["someone/a-language-model"]["loadable"] is False


def test_each_view_lists_its_formats(catalog: Catalog) -> None:
    """Bug: a view lists the wrong formats, so a client sees language models among the image models, or an hf
    repository with only its small files (no weights) among the models."""
    cache_repo("local/m", with_sidecar=True)
    cache_repo("Tongyi-MAI/Z-Image-Turbo", mflux=False, size_gb=2)
    cache_repo("Tongyi-MAI/Z-Image", mflux=False)  # small files only
    cache_repo("someone/llm", mflux=False)

    def ids(view: str) -> list[str]:
        return [m["id"] for m in catalog.list_models(view, {})["models"]]

    assert ids("models") == ["Tongyi-MAI/Z-Image-Turbo", "local/m"]
    assert ids("mflux") == ["local/m"]
    assert ids("hf") == ["Tongyi-MAI/Z-Image-Turbo"]
    assert ids("hf/all") == ["Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"]
    assert ids("other") == ["someone/llm"]
    assert sorted(ids("all")) == ["Tongyi-MAI/Z-Image-Turbo", "local/m", "someone/llm"]


def test_a_rescan_replaces_the_rows(catalog: Catalog) -> None:
    """Bug: a deleted model stays in the list because a scan only adds rows."""
    import shutil

    cache_repo("local/gone", with_sidecar=True)
    assert len(catalog.list_models("mflux", {})["models"]) == 1
    shutil.rmtree(hf_models.repo_folder("local/gone"))
    catalog.scan()
    assert catalog.list_models("mflux", {})["models"] == []


def test_a_bad_view_cursor_or_limit_is_a_client_error_not_a_500(catalog: Catalog) -> None:
    """Bug: cursor=abc raises ValueError, and the route answers with a bare 500."""
    from mlx_imagine.errors import ApiError

    for bad in ({"cursor": "abc"}, {"cursor": "-1"}, {"limit": "x"}):
        with pytest.raises(ApiError) as caught:
            catalog.list_models("models", bad)
        assert caught.value.status == 400
    with pytest.raises(ApiError) as caught:
        catalog.list_models("remote", {})
    assert caught.value.status == 404


def test_the_manifest_sync_does_not_reach_the_network_when_hf_access_is_off(catalog: Catalog, monkeypatch) -> None:
    """Bug: allow_hf_access is false, and the cron thread still downloads the manifest."""
    monkeypatch.setattr(catalog_module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network"))
    catalog.settings.hf_allow_access = False
    assert catalog.sync_manifest() == "offline"


def test_the_list_routes_and_the_model_route(engine: Engine, catalog: Catalog) -> None:
    """Bug: a list route ignores a filter, the model route does not find a repository, or an old route still
    answers."""
    cache_repo("local/m", with_sidecar=True)
    cache_repo("someone/llm", mflux=False)
    client = TestClient(create_app(engine))
    assert [m["id"] for m in client.get("/v8/models").json()["models"]] == ["local/m"]
    assert [m["id"] for m in client.get("/v8/models/other").json()["models"]] == ["someone/llm"]
    assert [m["id"] for m in client.get("/v8/models/all", params={"q": "LLM"}).json()["models"]] == ["someone/llm"]
    assert client.get("/v8/models/local/m").json()["format"] == "mflux"
    assert client.get("/v8/models/local/none").status_code == 404
    assert client.get("/v8/models/nope").status_code == 404
    for old in ("/v8/model", "/v8/saved", "/v8/models/mflux_dir/m/x"):
        assert client.get(old).status_code == 404


def test_the_base_model_route_writes_the_sidecar(engine: Engine, catalog: Catalog) -> None:
    """Bug: PATCH does not record the base model, so the model stays not loadable."""
    snapshot = cache_repo("local/m")
    client = TestClient(create_app(engine))
    assert client.get("/v8/models/local/m").json()["loadable"] is False
    response = client.patch("/v8/models/local/m", json={"base_model": "z-image-turbo"})
    assert response.status_code == 200 and response.json()["loadable"] is True
    assert saved_models.read_sidecar(snapshot)["base_model"] == "z-image-turbo"
    cache_repo("someone/llm", mflux=False)
    assert client.patch("/v8/models/someone/llm", json={"base_model": "z-image-turbo"}).status_code == 400


def _fake_download(monkeypatch) -> None:
    def download_repo(repo_id: str, job) -> Path:
        return cache_repo(repo_id)

    monkeypatch.setattr("mlx_imagine.hf_models.download_repo", download_repo)


def test_a_community_model_downloads_into_the_hf_cache(engine: Engine, catalog: Catalog, monkeypatch) -> None:
    """Bug: a model of the community manifest does not reach the cache, or does not load with its base model."""

    from mlx_imagine.errors import ApiError

    catalog.load_manifest()
    _fake_download(monkeypatch)
    repo = "mflux-community/z-image-turbo-mflux-q8"
    job = wait_for(engine.submit_download(repo))
    assert job.state == "succeeded", job.error
    assert job.result["format"] == "mflux" and job.result["loadable"] is True
    assert catalog.get(repo)["base_model"] == "z-image-turbo"
    with pytest.raises(ApiError) as caught:
        engine.submit_download(repo)
    assert caught.value.code == "name_in_use"


def test_a_community_download_follows_the_download_settings(engine: Engine, catalog: Catalog, monkeypatch) -> None:
    """Bug: a community download ignores the weight switch, the confirmation or the model limit."""
    from mlx_imagine.errors import ApiError

    catalog.load_manifest()
    _fake_download(monkeypatch)
    monkeypatch.setattr("mlx_imagine.hf_models.estimate_repo_bytes", lambda repo_id: 4_200_000_000)
    repo = "mflux-community/z-image-turbo-mflux-q4"
    engine.settings.hf_allow_model_download = False
    with pytest.raises(ApiError) as caught:
        engine.submit_download(repo)
    assert caught.value.status == 403
    engine.settings.hf_allow_model_download = True
    engine.settings.hf_require_model_confirmation = True
    with pytest.raises(ApiError) as caught:
        engine.submit_download(repo)
    assert caught.value.code == "confirm_required" and "4.2 GB" in caught.value.message
    engine.settings.hf_max_models = 0
    with pytest.raises(ApiError) as caught:
        engine.submit_download(repo, confirm=True)
    assert caught.value.code == "limit_reached"


def test_a_community_download_checks_the_size_limit_with_the_incoming_model(
    engine: Engine, catalog: Catalog, monkeypatch
) -> None:
    """Bug: max_size_gb counts only the models on disk, so one 30 GB download passes a 1 GB limit."""

    from mlx_imagine.errors import ApiError

    catalog.load_manifest()
    _fake_download(monkeypatch)
    engine.settings.hf_max_size_gb = 1
    monkeypatch.setattr("mlx_imagine.hf_models.estimate_repo_bytes", lambda repo_id: 30 * 10**9)
    with pytest.raises(ApiError) as caught:
        engine.submit_download("mflux-community/z-image-turbo-mflux-q8")
    assert caught.value.code == "limit_reached"
    monkeypatch.setattr("mlx_imagine.hf_models.estimate_repo_bytes", lambda repo_id: None)  # the Hub does not answer
    monkeypatch.setattr("mlx_imagine.saved_models._size", lambda path: 5 * 10**9)
    job = wait_for(engine.submit_download("mflux-community/z-image-turbo-mflux-q8"))
    assert job.state == "failed" and job.error["code"] == "limit_reached"
    assert not hf_models.repo_folder("mflux-community/z-image-turbo-mflux-q8").exists()


def test_a_delete_of_a_symlinked_alias_keeps_the_files_it_points_to(engine: Engine, tmp_path: Path) -> None:
    """Bug: the delete follows the links of an alias repository and deletes the model folder outside the cache."""
    outside = tmp_path / "MFlux-Models" / "m"
    (outside / "transformer").mkdir(parents=True)
    (outside / "transformer" / saved_models.INDEX).write_text('{"metadata": {"quantization_level": "8"}}')
    folder = hf_models.repo_folder("local/m")
    snapshot = folder / "snapshots" / ("b" * 40) / "transformer"
    snapshot.mkdir(parents=True)
    (snapshot / saved_models.INDEX).symlink_to(outside / "transformer" / saved_models.INDEX)
    result = engine.delete_model("local/m", confirm=True)
    assert result["repo_id"] == "local/m" and not folder.exists()
    assert (outside / "transformer" / saved_models.INDEX).is_file()
