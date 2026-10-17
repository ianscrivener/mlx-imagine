"""The real registry: which mflux models the server runs, and through which command."""

import pytest

from mlx_imagine.errors import ApiError
from mlx_imagine.registry import REASON_NO_STEPS, Registry


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry()


def test_the_stage_1_models_and_their_commands(registry: Registry) -> None:
    """Bug: a model goes to the wrong command (z-image-turbo through mflux-generate-z-image, which also accepts it),
    or a stage 1 command loses its models."""
    supported = {entry.name: entry.spec.command for entry in registry.entries().values() if entry.supported}
    assert supported["z-image-turbo"] == "mflux-generate-z-image-turbo"
    assert supported["z-image"] == "mflux-generate-z-image"
    assert supported["qwen-image-edit"] == "mflux-generate-qwen-edit"
    assert supported["ideogram-4-fp8"] == "mflux-generate-ideogram4"
    assert {spec for spec in supported.values()} == {
        "mflux-generate-z-image-turbo", "mflux-generate-z-image", "mflux-generate-ernie-image-turbo",
        "mflux-generate-ernie-image", "mflux-generate-krea2", "mflux-generate-lens", "mflux-generate-boogu",
        "mflux-generate-ideogram4", "mflux-generate-qwen", "mflux-generate-qwen-edit", "mflux-generate-flux3",
    }  # fmt: skip


def test_the_flux3_models_run_through_the_adapter(registry: Registry) -> None:
    """Bug: a FLUX 3 model shows as unsupported, or the adapter takes a model that is not FLUX 3 Image Klein."""
    for name in ("flux3-image-klein", "flux3-image-klein-base"):
        entry = registry.find(name)
        assert entry.supported and entry.spec.module == "mlx_imagine.adapters.flux3"
    assert registry.find("flux2-klein-4b").spec is None


def test_models_without_steps_or_in_stage_2_are_listed_with_the_reason(registry: Registry) -> None:
    """Bug: FLUX or a ControlNet model shows as supported, so a load fails deep inside mflux."""
    dev = registry.find("dev")
    assert not dev.supported and dev.reason == REASON_NO_STEPS
    controlnet = registry.find("z-image-turbo-controlnet-union-2.1")
    assert not controlnet.supported and controlnet.reason == REASON_NO_STEPS
    with pytest.raises(ApiError) as caught:
        registry.supported("dev")
    assert caught.value.code == "model_not_supported"


def test_a_model_is_found_by_alias_and_by_repository_id(registry: Registry) -> None:
    """Bug: a client that sends an alias or a repo ID gets not_found."""
    assert registry.find("zimage-turbo").name == "z-image-turbo"
    assert registry.find("Tongyi-MAI/Z-Image-Turbo").name == "z-image-turbo"
    with pytest.raises(ApiError) as caught:
        registry.find("no-such-model")
    assert caught.value.code == "not_found"


def test_each_supported_model_has_download_patterns(registry: Registry) -> None:
    """Bug: a download job fetches the whole repository, or fails, because a weight definition path is wrong."""
    for entry in registry.entries().values():
        if entry.supported:
            assert entry.spec.download_patterns(entry.name), entry.name
