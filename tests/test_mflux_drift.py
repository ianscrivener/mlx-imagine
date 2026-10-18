"""Pins the parts of mflux the server calls or copies that carry no stability promise.

When one fails after an mflux update: read mflux's diff of that function, bring the server in line with it,
then record the digest the failure message prints. Recorded against mflux 0.23.0 (PyPI)."""

import inspect

import pytest
from _fingerprint import ast_fingerprint
from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.common.cli import save
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.flux2.cli import flux2_generate
from mflux.models.flux3.cli import flux3_generate
from mflux.models.qwen21.cli import qwen21_controlnet_generate, qwen21_generate
from mflux.models.z_image.cli import z_image_turbo_generate_controlnet
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

from mlx_imagine.registry import ADAPTER_COMMANDS, COMMANDS

PINNED = {
    # engine.build_model_for_save copies mflux-save's main() from step 1 on. See NOTICE.
    "save.main": (save.main, "3205d015789d2262"),
    # adapters/flux3.py copies the statements of this main(). See NOTICE.
    "flux2_generate.main": (flux2_generate.main, "98b33e79c5f6572d"),
    "flux3_generate.main": (flux3_generate.main, "7d123c7e6603e472"),
    # adapters/qwen21.py copies the statements of this main().
    "qwen21_generate.main": (qwen21_generate.main, "d8ade010d7dbfd78"),
    # adapters/qwen21_controlnet.py copies the statements of this main().
    "qwen21_controlnet_generate.main": (qwen21_controlnet_generate.main, "c4f70aeb80940255"),
    # adapters/zimage_controlnet.py copies the statements of this main().
    "z_image_turbo_generate_controlnet.main": (z_image_turbo_generate_controlnet.main, "2b3d1e8a42120036"),
    # engine.save_image calls these two private functions to write EXIF and the metadata dictionary.
    "ImageUtil._embed_metadata": (ImageUtil._embed_metadata, "49ba7e90356b7792"),
    "GeneratedImage._get_metadata": (GeneratedImage._get_metadata, "511107ee1fe28b99"),
    # engine._do_generate calls these two private helpers instead of register_callbacks, whose memory saver
    # frees the text encoder of a model the server keeps loaded.
    "CallbackManager._apply_vae_tiling": (CallbackManager._apply_vae_tiling, "c763f7e8bd1be8d3"),
    "CallbackManager._register_memory_saver": (CallbackManager._register_memory_saver, "0732a64b6cb2e36f"),
    # The parser reads sys.argv here, which is why registry.parse_command_line swaps sys.argv under a lock.
    "CommandLineParser._option_was_provided": (CommandLineParser._option_was_provided, "cacf69061c4f6383"),
    # hf_models.is_complete uses it to tell whether a load can start without a download.
    "PathResolution._find_complete_cached_snapshot": (
        PathResolution._find_complete_cached_snapshot,
        "d385c87d0ee3cf1d",
    ),
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_the_pinned_mflux_function_is_unchanged(name: str) -> None:
    """Bug: mflux changes a private helper or the save flow the server depends on, and the server keeps calling or
    copying the old behavior."""
    function, digest = PINNED[name]
    actual = ast_fingerprint(function)
    assert actual == digest, f"{name} changed in mflux: review it, then record {actual}"


@pytest.mark.parametrize("spec", COMMANDS + ADAPTER_COMMANDS, ids=lambda spec: spec.command)
def test_each_stage_1_command_keeps_the_step_signatures_the_server_calls(spec) -> None:
    """Bug: mflux renames or reorders a command step parameter, and the engine's positional
    generate(model, args, seed, prompt) call binds the wrong values."""
    steps = spec.steps()
    assert list(inspect.signature(steps.validate).parameters) == ["args"]
    assert list(inspect.signature(steps.load).parameters) == ["args"]
    assert list(inspect.signature(steps.generate).parameters) == ["model", "args", "seed", "prompt"]


def test_generated_image_save_keeps_its_keywords() -> None:
    """Bug: GeneratedImage.save drops overwrite or export_json_metadata, and every generate job fails at save."""
    assert {"path", "export_json_metadata", "overwrite"} <= inspect.signature(GeneratedImage.save).parameters.keys()
