"""The engine with the fake z-image-turbo command: load, generate, low-RAM, cancel, save and delete rules."""

from pathlib import Path
from typing import Any

import pytest
from _fake_command import FakeModel
from conftest import wait_for

from mlx_imagine import saved_models
from mlx_imagine.engine import Engine
from mlx_imagine.errors import ApiError


def _load(engine: Engine, **request) -> None:
    job = wait_for(engine.submit_load({"model": "z-image-turbo", **request}))
    assert job.state == "succeeded", job.error


def test_load_then_generate_saves_one_image_per_seed(engine: Engine) -> None:
    """Bug: the model loads again for each image, or a seed's image is not saved in the output directory."""
    _load(engine)
    job = wait_for(engine.submit_generate({"prompt": "a cat", "seed": [1, 2], "steps": 3}))
    assert job.state == "succeeded", job.error
    assert len(FakeModel.instances) == 1
    images = job.result["images"]
    assert [image["seed"] for image in images] == [1, 2]
    for image in images:
        path = Path(image["path"])
        assert path.parent == engine.output_dir and path.is_file()
        assert path.with_suffix(".json").is_file()
    assert job.progress == {"image_index": 1, "image_count": 2, "step": 3, "total_steps": 3}


def test_generate_is_checked_against_the_target_model_before_it_waits(engine: Engine) -> None:
    """Bug: a generate request with no model, or with a bad option, waits in the queue and fails minutes later."""
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "a cat"})
    assert caught.value.code == "model_not_loaded"
    engine.submit_load({"model": "z-image-turbo"})  # still waiting: the target model is enough
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "a cat", "steps": "many"})
    assert caught.value.code == "option_invalid"
    assert engine.submit_generate({"prompt": "a cat", "steps": 1}).state in {"queued", "running", "succeeded"}


def test_a_load_request_for_a_model_the_command_refuses_fails_at_once(engine: Engine) -> None:
    """Bug: the server accepts a model its command cannot run, and the load fails inside mflux."""
    with pytest.raises(ApiError) as caught:
        engine.submit_load({"model": "dev"})
    assert caught.value.code == "model_not_supported"


def test_the_prompt_is_used_as_given_and_never_read_from_stdin(engine: Engine) -> None:
    """Bug: the server calls PromptUtil.read_prompt, so the prompt "-" reads stdin, which is the MCP stdio channel."""
    _load(engine)
    job = wait_for(engine.submit_generate({"prompt": "-", "steps": 1}))
    assert job.state == "succeeded", job.error
    assert FakeModel.instances[0].generate_calls[-1]["prompt"] == "-"


def test_low_ram_mode_loads_the_model_again_before_the_next_job(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: after a low-RAM run freed the transformer, the next job uses the broken model."""
    from mflux.callbacks.callback_manager import CallbackManager

    registered = []

    def fake_memory_saver(args, model):
        saver = type("Saver", (), {"call_in_loop": lambda self, **kwargs: None})()
        model.callbacks.register(saver)
        registered.append(saver)
        return saver

    monkeypatch.setattr(CallbackManager, "_register_memory_saver", staticmethod(fake_memory_saver))
    _load(engine, low_ram=True)
    for _ in range(2):
        assert wait_for(engine.submit_generate({"prompt": "a cat", "steps": 1})).state == "succeeded"
    assert len(FakeModel.instances) == 2  # one load, one reload before the second job
    assert engine.loaded.state == "needs_reload"
    assert registered[0] not in FakeModel.instances[0].callbacks.in_loop


def test_a_cancel_during_a_run_stops_it_and_keeps_the_images_that_are_complete(engine: Engine) -> None:
    """Bug: a cancel waits for all seeds, or it drops the images that were already saved."""
    _load(engine)
    FakeModel.hold_at = (2, 1)
    job = engine.submit_generate({"prompt": "a cat", "seed": [1, 2, 3], "steps": 2})
    assert FakeModel.held.wait(5)  # the first image is complete, the second waits at step 1
    engine.queue.cancel(job.id)
    FakeModel.release.set()
    wait_for(job)
    assert job.state == "cancelled"
    assert [image["seed"] for image in job.result["images"]] == [1]


def test_unload_frees_the_model_and_clears_the_prompt_cache(engine: Engine) -> None:
    """Bug: unload keeps a reference to the model or its prompt cache, so the memory is never freed."""
    _load(engine)
    wait_for(engine.submit_generate({"prompt": "a cat", "steps": 1}))
    model = FakeModel.instances[0]
    assert model.prompt_cache
    assert wait_for(engine.submit_unload()).state == "succeeded"
    assert engine.loaded is None and engine.target is None
    assert model.prompt_cache == {}


def test_a_failed_load_resets_the_target_to_the_model_that_is_loaded(engine: Engine, monkeypatch) -> None:
    """Bug: after a failed load, generate requests are checked against a model that never loaded."""
    from _fake_command import FakeCommand

    def broken(args):
        raise RuntimeError("weights missing")

    monkeypatch.setattr(FakeCommand, "load", staticmethod(broken))
    job = wait_for(engine.submit_load({"model": "z-image-turbo"}))
    assert job.state == "failed"
    assert engine.target is None


def test_a_saved_model_loads_by_name_with_its_sidecar(engine: Engine) -> None:
    """Bug: a saved model loads without --base-model, so mflux cannot tell its architecture."""
    path = engine.models_dir / "zit-q8"
    (path / "transformer").mkdir(parents=True)
    (path / "transformer" / saved_models.INDEX).write_text('{"metadata": {"quantization_level": "8"}}')
    with pytest.raises(ApiError) as caught:
        engine.submit_load({"model": "zit-q8"})
    assert caught.value.code == "base_model_required"
    first = wait_for(engine.submit_load({"model": "zit-q8", "base_model": "z-image-turbo"}))
    assert first.state == "succeeded", first.error
    assert saved_models.read_sidecar(path)["base_model"] == "z-image-turbo"  # the load wrote the sidecar
    _load_job = wait_for(engine.submit_load({"model": "zit-q8"}))
    assert _load_job.state == "succeeded", _load_job.error
    assert FakeModel.instances[-1].args.model_path == str(path)
    with pytest.raises(ApiError) as caught:
        engine.delete_saved_model("zit-q8", confirm=True)
    assert caught.value.code == "model_in_use"


def test_deletes_need_confirm(engine: Engine) -> None:
    """Bug: a delete without confirm removes a model of many gigabytes."""
    with pytest.raises(ApiError) as caught:
        engine.delete_saved_model("anything", confirm=False)
    assert caught.value.code == "confirm_required"
    with pytest.raises(ApiError) as caught:
        engine.delete_hf_model("Tongyi-MAI/Z-Image-Turbo", confirm=False)
    assert caught.value.code == "confirm_required"


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", ""])
def test_a_save_name_must_be_one_directory_name(engine: Engine, name: str) -> None:
    """Bug: a save name with .. or / writes the model outside the model directory."""
    with pytest.raises(ApiError) as caught:
        engine.submit_save({"model": "z-image-turbo", "quantize": 8, "name": name})
    assert caught.value.code == "invalid_request"


def test_image_paths_outside_the_input_directory_are_refused(engine: Engine) -> None:
    """Bug: image_path reads any file on the machine."""
    _load(engine)
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "a cat", "image_path": "/etc/hosts"})
    assert caught.value.code == "path_not_allowed"


def test_materialize_reads_the_weights_of_a_model_or_of_its_parts() -> None:
    """Bug: a load job ends before any weight is read, so the first generate job pays the load and an
    out-of-memory failure happens mid-generate."""
    import mlx.core as mx
    import mlx.nn as nn

    from mlx_imagine.engine import materialize

    class Holder:  # a model class that is not an nn.Module, like LensImage
        def __init__(self) -> None:
            self.transformer = nn.Linear(4, 4)
            self.name = "not a module"

    for model in (nn.Linear(4, 4), Holder()):
        mx.clear_cache()
        materialize(model)  # must not raise on attributes that are not modules
    materialize(object())  # a fake model with no parts


def test_a_stage_2_model_download_is_refused_before_it_waits(engine: Engine, monkeypatch) -> None:
    """Bug: a stage 2 model passes the `spec is None` check, waits in the queue, then fails with a ValueError
    from its empty weight definition."""
    from mlx_imagine.errors import ApiError as Error
    from mlx_imagine.registry import REASON_STAGE_2, CommandSpec, ModelEntry

    stage_2 = ModelEntry(
        "cn", (), "owner/cn", CommandSpec("x", "cmd", "mod", "Cls", "", stage=2), reason=REASON_STAGE_2
    )
    monkeypatch.setitem(engine.registry.entries(), "cn", stage_2)
    with pytest.raises(Error) as caught:
        engine.submit_download("cn")
    assert caught.value.code == "model_not_supported"


def test_a_save_request_clears_the_target_model(engine: Engine) -> None:
    """Bug: a save job unloads the model, but generate requests queued after it are checked against the old
    target, then fail with model_not_loaded when they run."""
    _load(engine)
    FakeModel.hold_at = None
    engine.queue.stop(timeout=5)  # keep the save job waiting
    engine.submit_save({"model": "z-image-turbo", "quantize": 8, "name": "kept-waiting"})
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "a cat"})
    assert caught.value.code == "model_not_loaded"


def test_a_failed_save_leaves_no_partial_folder(engine: Engine, monkeypatch) -> None:
    """Bug: a save that fails leaves `.<name>.partial`, which the API cannot delete and which counts as a
    saved model."""
    from mlx_imagine import engine as engine_module

    def broken(model_name, quantize):
        (engine.models_dir / ".broken.partial" / "transformer").mkdir(parents=True)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(engine_module, "build_model_for_save", broken)
    job = wait_for(engine.submit_save({"model": "z-image-turbo", "quantize": 8, "name": "broken"}))
    assert job.state == "failed"
    assert not (engine.models_dir / ".broken.partial").exists()
    assert saved_models.list_models(engine.models_dir) == []


def test_image_paths_must_be_strings(engine: Engine) -> None:
    """Bug: `"image_paths": [123]` reaches value.startswith() and returns a bare 500."""
    _load(engine)
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "a cat", "image_paths": [123]})
    assert caught.value.code == "invalid_request"


def test_an_mflux_format_model_in_the_hf_cache_loads_by_repo_id(engine: Engine) -> None:
    """Bug: an mflux-format repository in the Hugging Face cache cannot load, so the user must copy it into the
    mflux model directory first."""
    import os

    snapshot = Path(os.environ["HF_HUB_CACHE"]) / "models--mflux-community--zit-q8" / "snapshots" / "abc123"
    (snapshot / "transformer").mkdir(parents=True)
    (snapshot / "transformer" / saved_models.INDEX).write_text('{"metadata": {"quantization_level": "8"}}')
    with pytest.raises(ApiError) as caught:
        engine.submit_load({"model": "mflux-community/zit-q8"})
    assert caught.value.code == "base_model_required"
    with pytest.raises(ApiError) as caught:
        engine.submit_load({"model": "mflux-community/zit-q8", "base_model": "z-image-turbo", "quantize": 4})
    assert caught.value.code == "invalid_request"
    job = wait_for(engine.submit_load({"model": "mflux-community/zit-q8", "base_model": "z-image-turbo"}))
    assert job.state == "succeeded", job.error
    assert engine.loaded.spec.hub_repo == "mflux-community/zit-q8"
    assert Path(FakeModel.instances[-1].args.model_path).resolve() == snapshot.resolve()
    with pytest.raises(ApiError) as caught:
        engine.delete_hf_model("mflux-community/zit-q8", confirm=True)
    assert caught.value.code == "model_in_use"


def _generate_and_open(engine: Engine, **request: Any) -> dict[str, Any]:
    import PIL.Image

    wait_for(engine.submit_load({"model": "z-image-turbo"}))
    job = wait_for(engine.submit_generate({"prompt": "a cat", "steps": 1, "seed": [3], **request}))
    assert job.state == "succeeded", job.error
    with PIL.Image.open(job.result["images"][0]["path"]) as image:
        image.load()
        return dict(image.info)


def test_the_metadata_settings_and_the_request_decide_what_the_png_holds(engine: Engine) -> None:
    """Bug: exif_enabled or iptc_enabled is ignored, so a user who turned metadata off still gets the prompt in
    the file, or a request cannot override the setting."""
    both = _generate_and_open(engine)
    assert both.get("exif") and "XML:com.adobe.xmp" in both
    only_iptc = _generate_and_open(engine, exif_enabled=False)
    assert not only_iptc.get("exif") and "XML:com.adobe.xmp" in only_iptc
    engine.settings.iptc_enabled = False
    only_exif = _generate_and_open(engine)
    assert only_exif.get("exif") and "XML:com.adobe.xmp" not in only_exif
    neither = _generate_and_open(engine, exif_enabled=False)
    assert not neither.get("exif") and "XML:com.adobe.xmp" not in neither
    back_on = _generate_and_open(engine, iptc_enabled=True)  # a request value wins over the setting
    assert "XML:com.adobe.xmp" in back_on


def test_the_json_sidecar_follows_the_setting_and_the_request(engine: Engine) -> None:
    """Bug: json_sidecar_enabled is ignored, and a .metadata.json file still appears next to each image."""
    import json

    def sidecars(**request: Any) -> list[Path]:
        wait_for(engine.submit_load({"model": "z-image-turbo"}))
        job = wait_for(engine.submit_generate({"prompt": "a cat", "steps": 1, "seed": [4], **request}))
        assert job.state == "succeeded", job.error
        return list(Path(job.result["images"][0]["path"]).parent.glob(f"{job.id}*.json"))

    assert len(sidecars()) == 1
    assert "prompt" in json.loads(sidecars()[0].read_text())
    assert sidecars(json_sidecar_enabled=False) == []
    engine.settings.json_sidecar_enabled = False
    assert sidecars() == []
    assert len(sidecars(json_sidecar_enabled=True)) == 1


def test_a_metadata_option_must_be_a_boolean(engine: Engine) -> None:
    """Bug: exif_enabled: "no" counts as true, and the file keeps its metadata."""
    wait_for(engine.submit_load({"model": "z-image-turbo"}))
    with pytest.raises(ApiError) as caught:
        engine.submit_generate({"prompt": "x", "exif_enabled": "no"})
    assert caught.value.code == "option_invalid"


def test_an_option_that_the_model_ignores_is_refused(engine: Engine) -> None:
    """Bug: the server accepts guidance for a guidance-distilled model, mflux ignores it with no message, and the
    client thinks that the value had an effect."""
    wait_for(engine.submit_load({"model": "z-image-turbo"}))
    for name, value in (("guidance", 3.5), ("negative_prompt", "blurry")):
        with pytest.raises(ApiError) as caught:
            engine.submit_generate({"prompt": "x", name: value})
        assert caught.value.code == "option_ignored" and caught.value.details["option"] == name
    assert engine.submit_generate({"prompt": "x", "steps": 1}) is not None


def test_the_loaded_model_reports_its_capabilities(engine: Engine) -> None:
    """Bug: a client has no way to learn that the model ignores guidance, so it sends one and gets an error."""
    job = wait_for(engine.submit_load({"model": "z-image-turbo"}))
    assert job.state == "succeeded", job.error
    caps = engine.loaded.to_dict()["capabilities"]
    assert caps["supports_guidance"] is False and caps["supports_negative_prompt"] is False
    assert caps["default_steps"] and "txt2img" in caps["modes"]


def test_a_load_of_an_incomplete_model_fails_at_once_and_names_the_saved_copy(engine, monkeypatch) -> None:
    """Bug: the load job waits in the queue and then fails with "send a download request", and the client does not
    learn that a saved copy of the model is on this Mac."""
    from mlx_imagine import hf_models
    from mlx_imagine.errors import ApiError

    monkeypatch.setattr(hf_models, "is_complete", lambda entry: False)
    saved = engine.models_dir / "Tongyi-MAI--Z-Image-Turbo-mflux-q8"
    saved.mkdir(parents=True)
    (saved / "model.safetensors.index.json").write_text("{}")
    (saved / "transformer").mkdir()
    (saved / "transformer" / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ApiError) as error:
        engine.submit_load({"model": "z-image-turbo"})
    assert error.value.code == "not_found"
    assert "Tongyi-MAI--Z-Image-Turbo-mflux-q8" in error.value.message and "base_model" in error.value.message
    assert not engine.queue.waiting()
