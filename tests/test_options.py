"""Requests in mflux option names become a command line that mflux's own parser checks."""

import sys

import pytest
from _fake_command import FAKE_SPEC

from mlx_imagine.errors import ApiError
from mlx_imagine.options import generate_names, parse, schemas, to_argv


def _argv(request: dict, allowed: frozenset | None = None) -> list[str]:
    return to_argv(FAKE_SPEC.build_parser(), request, allowed=allowed)


def test_each_option_kind_becomes_the_flag_mflux_parses() -> None:
    """Bug: a boolean, a list or a negative number is passed as a bare string, so argparse reads the wrong value
    or reads the next token as the value."""
    argv = _argv({"steps": 4, "seed": [1, 2], "pid_decode": False, "vae_tiling": True, "prompt": "-a cat"})
    assert argv == ["--steps", "4", "--seed", "1", "2", "--no-pid-decode", "--vae-tiling", "--prompt=-a cat"]
    args, _ = parse(FAKE_SPEC, ["--model", "z-image-turbo", *argv])
    assert (args.steps, args.seed, args.pid_decode, args.vae_tiling, args.prompt) == (4, [1, 2], False, True, "-a cat")


@pytest.mark.parametrize(
    ("request_body", "code"),
    [
        ({"output": "/etc/x.png"}, "option_blocked"),
        ({"prompt_file": "/etc/passwd"}, "option_blocked"),
        ({"lora": [["a.safetensors"]]}, "option_blocked"),
        ({"no_such_option": 1}, "option_invalid"),
        ({"steps": True}, "option_invalid"),
        ({"seed": [{"x": 1}]}, "option_invalid"),
    ],
)
def test_unsafe_and_unknown_options_are_refused_before_the_parser_runs(request_body: dict, code: str) -> None:
    """Bug: the server passes --output or --prompt-file through, so a client writes or reads any file, or an
    unknown key is dropped without an error."""
    with pytest.raises(ApiError) as caught:
        _argv(request_body)
    assert caught.value.code == code
    assert caught.value.status == 422


def test_a_load_option_in_a_generate_request_names_the_load_request() -> None:
    """Bug: --quantize in a generate request is accepted and does nothing, because the model already loaded."""
    with pytest.raises(ApiError) as caught:
        _argv({"quantize": 4}, allowed=generate_names(FAKE_SPEC))
    assert caught.value.code == "option_for_load"


def test_a_parser_error_is_an_api_error_and_sys_argv_is_restored() -> None:
    """Bug: argparse calls sys.exit() inside the server, or the swapped sys.argv leaks into the process."""
    before = list(sys.argv)
    with pytest.raises(ApiError) as caught:
        parse(FAKE_SPEC, ["--model", "z-image-turbo", "--prompt", "x", "--steps", "many"])
    assert caught.value.code == "option_invalid"
    assert "--steps" in caught.value.message
    assert sys.argv == before


def test_the_generate_schema_comes_from_the_parser_without_blocked_or_load_options() -> None:
    """Bug: the schema lists --output or --quantize for generate, or loses prompt as a required field."""
    generated = schemas(FAKE_SPEC)
    generate, load = generated["generate_schema"], generated["load_schema"]
    assert generate["required"] == ["prompt"]
    assert generate["properties"]["seed"]["type"] == "array"
    assert generate["properties"]["steps"]["type"] == "integer"
    assert not {"output", "prompt_file", "lora", "quantize", "model"} & generate["properties"].keys()
    assert {"model", "quantize", "low_ram"} <= load["properties"].keys()
    assert load["properties"]["quantize"]["enum"] == [3, 4, 5, 6, 8]
    # z-image-turbo ignores --guidance (IGNORED_OPTIONS), so the schema does not offer it.
    assert "guidance" not in generate["properties"]
