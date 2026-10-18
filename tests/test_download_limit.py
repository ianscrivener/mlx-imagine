"""huggingface_models.max_download_model_size_gb limits one download: a model, or a LoRA file."""

import pytest
from conftest import wait_for

from mlx_imagine import hf_models, loras
from mlx_imagine.engine import GB, Engine
from mlx_imagine.errors import ApiError


def _refused(engine: Engine, **call) -> ApiError:
    with pytest.raises(ApiError) as error:
        engine.submit_lora_download("owner/name", "a.safetensors", **call)
    return error.value


def test_a_lora_larger_than_the_limit_is_refused_and_a_small_one_is_not(engine: Engine, monkeypatch) -> None:
    """Bug: a download of 1.3 GB starts although the owner set 0.4 GB as the most for one download."""
    engine.settings.hf_max_download_gb = 0.4
    engine.settings.hf_require_model_confirmation = False
    monkeypatch.setattr(loras, "estimate_bytes", lambda repo, name: int(1.3 * GB))
    error = _refused(engine)
    assert error.code == "limit_exceeded" and error.status == 422
    assert "1.30 GB" in error.message and "max_download_model_size_gb is 0.4" in error.message
    monkeypatch.setattr(loras, "estimate_bytes", lambda repo, name: int(0.1 * GB))
    monkeypatch.setattr(loras, "download", lambda *args: {"name": "x", "size_mb": 100.0})
    assert wait_for(engine.submit_lora_download("owner/name", "a.safetensors")).state == "succeeded"


def test_an_unknown_size_is_refused_when_the_limit_is_on_and_allowed_when_it_is_off(
    engine: Engine, monkeypatch
) -> None:
    """Bug: the limit is skipped when the Hub gives no size, so any file can be fetched."""
    engine.settings.hf_require_model_confirmation = False
    monkeypatch.setattr(loras, "estimate_bytes", lambda repo, name: None)
    engine.settings.hf_max_download_gb = 0.4
    assert "cannot get the size" in _refused(engine).message
    engine.settings.hf_max_download_gb = None
    monkeypatch.setattr(loras, "download", lambda *args: {"name": "x", "size_mb": 1.0})
    assert wait_for(engine.submit_lora_download("owner/name", "a.safetensors")).state == "succeeded"


def test_a_model_download_larger_than_the_limit_is_refused_before_the_confirmation(engine: Engine, monkeypatch) -> None:
    """Bug: the owner confirms a 33 GB download that the limit forbids, or the limit does not apply to models."""
    engine.settings.hf_max_download_gb = 0.4
    monkeypatch.setattr(hf_models, "estimate_bytes", lambda entry: 33 * GB)
    with pytest.raises(ApiError) as error:
        engine.submit_download("z-image-turbo", confirm=False)
    assert error.value.code == "limit_exceeded"
    monkeypatch.setattr(hf_models, "estimate_bytes", lambda entry: int(0.2 * GB))
    engine.settings.hf_require_model_confirmation = True
    with pytest.raises(ApiError) as needs:
        engine.submit_download("z-image-turbo", confirm=False)
    assert needs.value.code == "confirm_required"


def test_the_limit_shows_in_the_server_state_and_a_bad_value_is_refused(engine: Engine) -> None:
    """Bug: a client cannot see the limit before a refused call, or the setting accepts 0 or a negative size."""
    engine.settings.hf_max_download_gb = 0.4
    assert engine.status()["permissions"]["max_download_gb"] == 0.4
    for bad in (0, -1, "big"):
        with pytest.raises(ApiError):
            engine.update_settings({"huggingface_models": {"max_download_model_size_gb": bad}})
    engine.update_settings({"huggingface_models": {"max_download_model_size_gb": 2}})
    assert engine.settings.hf_max_download_gb == 2


def test_the_progress_hook_of_a_download_counts_and_reports(engine: Engine) -> None:
    """Bug: the progress class of a download raised AttributeError 'unit', so the job failed although the file
    was complete."""
    from mlx_imagine.jobs import Job

    job = Job("download", {}, lambda job: None)
    bar = hf_models._progress_class(job)(total=1000, unit="B", desc="a.safetensors")
    bar.update(400)
    assert job.progress["bytes_done"] == 400 and job.progress["bytes_total"] == 1000
