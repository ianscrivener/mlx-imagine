"""The adapters: commands for mflux models that have no command steps. No weights load."""

import inspect

import pytest
from mflux.utils.exceptions import ModelConfigError

from mlx_imagine.registry import ADAPTER_COMMANDS, parse_command_line


@pytest.mark.parametrize("spec", ADAPTER_COMMANDS, ids=lambda spec: spec.command)
def test_an_adapter_has_the_shape_of_an_mflux_command(spec) -> None:
    """Bug: an adapter misses a part of the command shape, and the registry or the engine fails on it at run time."""
    module = spec.load_module()
    assert callable(module.build_parser) and isinstance(module.DEFAULT_MODEL, str)
    assert isinstance(module.IGNORED_OPTIONS, dict)
    steps = spec.steps()
    assert list(inspect.signature(steps.validate).parameters) == ["args"]
    assert list(inspect.signature(steps.load).parameters) == ["args"]
    assert list(inspect.signature(steps.generate).parameters) == ["model", "args", "seed", "prompt"]


def test_flux3_validate_refuses_a_negative_prompt_and_a_foreign_model() -> None:
    """Bug: the adapter accepts a negative prompt that FLUX 3 Image Klein ignores, or a model of another family."""
    spec = next(spec for spec in ADAPTER_COMMANDS if spec.family == "flux3")
    parser = spec.build_parser()
    args = parse_command_line(parser, ["--prompt", "x", "--model", "flux3-image-klein"])
    assert spec.steps().validate(args).model_name
    with pytest.raises(ModelConfigError):
        spec.steps().validate(parse_command_line(parser, ["--prompt", "x", "--model", "z-image-turbo"]))
    with pytest.raises(ModelConfigError):
        spec.steps().validate(parse_command_line(parser, ["--prompt", "x", "--negative-prompt", "y"]))
