"""The Qwen-Image 2.1 ControlNet adapter: its checks, its modes and the image options of a request."""

import pytest
from mflux.utils.exceptions import ModelConfigError

from mlx_imagine.adapters import qwen21_controlnet as adapter
from mlx_imagine.errors import ApiError
from mlx_imagine.options import check_image_paths
from mlx_imagine.registry import Registry, parse_command_line, placeholder_argv

MODEL = ["--model", "qwen-image-2.1-controlnet-union", "--prompt", "x"]


def _validate(*options: str) -> None:
    args = parse_command_line(adapter.build_parser(), [*MODEL, *options])
    adapter.Qwen21ControlnetCommand.validate(args)


def test_the_two_controlnet_models_are_supported_and_have_the_two_modes() -> None:
    """Bug: the models stay unsupported, or a client sees the mode txt2img, which the command cannot do."""
    entries = Registry().entries()
    for name in ("qwen-image-2.1-controlnet-union", "qwen-image-2.1-turbo-controlnet-union"):
        assert entries[name].supported and entries[name].modes == ("controlnet", "inpaint")


def test_a_request_needs_a_control_image_or_an_image_and_a_mask() -> None:
    """Bug: validate() raises ValueError, which the server turns into HTTP 500, or accepts a request with no image."""
    with pytest.raises(ModelConfigError, match="controlnet_image_path"):
        _validate()
    with pytest.raises(ModelConfigError, match="both image_path and mask_image"):
        _validate("--image-path", "a.png")
    with pytest.raises(ModelConfigError, match="both image_path and mask_image"):
        _validate("--controlnet-image-path", "c.png", "--mask-image", "m.png")
    _validate("--controlnet-image-path", "c.png")
    _validate("--image-path", "a.png", "--mask-image", "m.png")


def test_the_load_check_uses_a_request_that_validate_accepts() -> None:
    """Bug: a load request fails with "Give controlnet_image_path", because the check has no image."""
    args = parse_command_line(adapter.build_parser(), [*MODEL[:2], *placeholder_argv(adapter.build_parser())])
    adapter.Qwen21ControlnetCommand.validate(args)


def test_the_turbo_model_refuses_guidance_above_one() -> None:
    """Bug: the Turbo model runs CFG that its checkpoint does not support."""
    args = parse_command_line(
        adapter.build_parser(),
        ["--model", "qwen-image-2.1-turbo-controlnet-union", "--prompt", "x", "--controlnet-image-path", "c.png",
         "--guidance", "3"],
    )  # fmt: skip
    with pytest.raises(ModelConfigError, match="without CFG"):
        adapter.Qwen21ControlnetCommand.validate(args)


def test_the_control_image_and_the_mask_resolve_inside_the_input_directory(tmp_path) -> None:
    """Bug: a client reads any file on the Mac through controlnet_image_path or mask_image."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "mask.png").write_bytes(b"x")

    def resolve(value: str):
        from mlx_imagine.paths import resolve_inside

        return resolve_inside(input_dir, value)

    checked = check_image_paths({"mask_image": "mask.png", "controlnet_image_path": "mask.png"}, input_dir, resolve)
    assert checked["mask_image"] == str((input_dir / "mask.png").resolve())
    with pytest.raises(ApiError):
        check_image_paths({"mask_image": "/etc/passwd"}, input_dir, resolve)
    with pytest.raises(ApiError):
        check_image_paths({"controlnet_image_path": "../x.png"}, input_dir, resolve)


def test_the_mask_preview_draws_the_white_area_in_red(engine, tmp_path) -> None:
    """Bug: the preview paints the black area, so a client checks the wrong region."""
    import io

    import PIL.Image
    from fastapi.testclient import TestClient

    from mlx_imagine.api import create_app

    engine.input_dir.mkdir(parents=True, exist_ok=True)
    PIL.Image.new("RGB", (64, 64), (0, 0, 255)).save(engine.input_dir / "a.png")
    mask = PIL.Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))  # the left half is drawn again
    mask.save(engine.input_dir / "m.png")
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/masks/preview", json={"image_path": "a.png", "mask_image": "m.png"})
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        shown = PIL.Image.open(io.BytesIO(response.content))
        assert shown.getpixel((8, 8))[0] > 100 and shown.getpixel((56, 8)) == (0, 0, 255)
        assert client.post("/v1/masks/preview", json={"image_path": "a.png"}).status_code == 400


def test_a_control_spec_resolves_its_path_inside_the_input_directory(tmp_path) -> None:
    """Bug: a client reads any file through `control`, or a strength at the end of the spec is lost."""
    from mlx_imagine.paths import resolve_inside

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "edges.png").write_bytes(b"x")

    def resolve(value: str):
        return resolve_inside(input_dir, value)

    path = str((input_dir / "edges.png").resolve())
    checked = check_image_paths({"control": ["canny:edges.png:0.8", "depth:edges.png"]}, input_dir, resolve)
    assert checked["control"] == [f"canny:{path}:0.8", f"depth:{path}"]
    for bad in ("canny", "canny:/etc/passwd", "canny:../x.png:0.5", 5):
        with pytest.raises(ApiError):
            check_image_paths({"control": [bad]}, input_dir, resolve)


def test_the_z_image_controlnet_needs_a_control_and_a_repeatable_option_becomes_flags() -> None:
    """Bug: `control` is not required in the schema, or a list of controls reaches argparse as one value."""
    from mlx_imagine.options import schemas, to_argv

    entry = Registry().supported("z-image-turbo-controlnet-union-2.1")
    generate = schemas(entry.spec)["generate_schema"]
    assert "control" in generate["required"] and generate["properties"]["control"]["type"] == "array"
    argv = to_argv(entry.spec.build_parser(), {"control": ["canny:a.png", "pose:b.png:0.7"]})
    assert argv == ["--control", "canny:a.png", "--control", "pose:b.png:0.7"]
    from mlx_imagine.adapters import zimage_controlnet as adapter

    with pytest.raises(ModelConfigError, match="Invalid --control"):
        args = parse_command_line(adapter.build_parser(), ["--prompt", "x", "--control", "nonsense"])
        adapter.ZImageTurboControlnetCommand.validate(args)
