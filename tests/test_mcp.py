"""The MCP tools: names, annotations, generated schemas and one generate call over the fake command."""

import asyncio
import json

import pytest

from mlx_imagine.engine import Engine
from mlx_imagine.mcp_server import build_mcp


def _tools(engine: Engine) -> dict:
    return {tool.name: tool for tool in asyncio.run(build_mcp(engine).list_tools())}


def test_each_operation_has_a_tool_and_the_deletes_are_marked_destructive(engine: Engine) -> None:
    """Bug: a delete tool has no destructive hint, so an agent client runs it without asking the user."""
    tools = _tools(engine)
    assert {"load_model", "unload_model", "download_model", "save_model", "get_job", "cancel_job"} <= tools.keys()
    for name in ("delete_model",):
        assert tools[name].annotations.destructive_hint is True
        assert "confirm" in tools[name].input_schema["required"]
    assert tools["list_models"].annotations.read_only_hint is True


def test_a_generate_tool_has_the_model_schema_and_no_context_parameter(engine: Engine) -> None:
    """Bug: the Context parameter shows in the tool schema, or the tool offers options the server blocks."""
    schema = _tools(engine)["generate_z_image_turbo"].input_schema
    assert "ctx" not in schema["properties"]
    assert schema["required"] == ["prompt"]
    assert {"steps", "seed", "wait_seconds"} <= schema["properties"].keys()
    assert not {"output", "quantize", "lora"} & schema["properties"].keys()


def test_a_generate_call_loads_the_model_and_returns_the_image(engine: Engine) -> None:
    """Bug: the tool returns the job ID only, though the image was ready within wait_seconds."""
    mcp = build_mcp(engine)
    result = asyncio.run(mcp.call_tool("generate_z_image_turbo", {"prompt": "a cat", "steps": 1, "wait_seconds": 10}))
    content = result.content if hasattr(result, "content") else result[0]
    job = json.loads(content[0].text)
    assert job["state"] == "succeeded", job["error"]
    assert content[1].type == "image" and content[1].mime_type == "image/png"


def test_set_config_changes_the_live_queue_like_the_rest_route(engine: Engine) -> None:
    """Bug: the MCP tool saves queue.queue_max_length and reports it as applied, but the running queue keeps the old
    limit, because only the REST route updated it."""
    mcp = build_mcp(engine)
    asyncio.run(mcp.call_tool("set_config", {"changes": {"queue": {"queue_max_length": 1, "job_history": 5}}}))
    assert engine.queue.max_length == 1
    assert engine.queue._history.maxlen == 5


def test_a_slow_generate_returns_the_job_and_a_get_job_hint_within_the_wait_cap(engine: Engine) -> None:
    """Bug: the generate tool waits up to 300 seconds, the host ends the call at about 60, and the agent gets an
    error although the job runs on."""
    from _fake_command import FakeModel

    from mlx_imagine.mcp_server import DEFAULT_GENERATE_WAIT, MAX_WAIT

    assert DEFAULT_GENERATE_WAIT < 60 and MAX_WAIT < 60
    schema = _tools(engine)["generate_z_image_turbo"].input_schema
    assert schema["properties"]["wait_seconds"]["default"] == DEFAULT_GENERATE_WAIT
    FakeModel.hold_at = (0, 0)
    mcp = build_mcp(engine)
    try:
        result = asyncio.run(
            mcp.call_tool("generate_z_image_turbo", {"prompt": "a cat", "steps": 2, "seed": [0], "wait_seconds": 1})
        )
        content = result.content if hasattr(result, "content") else result[0]
        job = json.loads(content[0].text)
        assert job["state"] in ("queued", "running")
        assert "get_job" in content[1].text and job["id"] in content[1].text
    finally:
        FakeModel.release.set()


def _call(mcp, name: str, arguments: dict):
    result = asyncio.run(mcp.call_tool(name, arguments))
    content = result.content if hasattr(result, "content") else result[0]
    return json.loads(content[0].text)


def test_the_lora_tools_list_delete_and_load(engine: Engine) -> None:
    """Bug: an MCP tool for LoRA is missing, a delete runs without confirm, or load_model drops `loras`."""
    tools = _tools(engine)
    assert {"list_loras", "download_lora", "delete_lora"} <= tools.keys()
    assert tools["delete_lora"].annotations.destructive_hint is True
    assert "confirm" in tools["delete_lora"].input_schema["required"]
    assert tools["list_loras"].annotations.read_only_hint is True
    assert "loras" in tools["load_model"].input_schema["properties"]

    path = engine.lora_dir / "style.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    mcp = build_mcp(engine)
    listed = asyncio.run(mcp.call_tool("list_loras", {}))
    assert "style.safetensors" in json.dumps(
        listed.structured_content if hasattr(listed, "structured_content") else listed
    )

    job = _call(
        mcp,
        "load_model",
        {"model": "z-image-turbo", "loras": [{"name": "style.safetensors", "scale": 0.5}], "wait_seconds": 10},
    )
    assert job["state"] == "succeeded", job["error"]
    assert engine.loaded.spec.loras == [{"name": "style.safetensors", "scale": 0.5}]

    with pytest.raises(Exception, match="lora_in_use"):
        asyncio.run(mcp.call_tool("delete_lora", {"name": "style.safetensors", "confirm": True}))
    assert path.exists()


def test_the_mask_preview_tool_is_read_only_and_returns_an_image(engine: Engine) -> None:
    """Bug: preview_mask changes state, or returns text, so an agent cannot see the mask."""
    import PIL.Image

    tool = _tools(engine)["preview_mask"]
    assert tool.annotations.read_only_hint is True
    engine.input_dir.mkdir(parents=True, exist_ok=True)
    PIL.Image.new("RGB", (32, 32)).save(engine.input_dir / "a.png")
    PIL.Image.new("L", (32, 32), 255).save(engine.input_dir / "m.png")
    result = asyncio.run(build_mcp(engine).call_tool("preview_mask", {"image_path": "a.png", "mask_image": "m.png"}))
    content = result.content if hasattr(result, "content") else result[0]
    assert content[0].type == "image" and content[0].mime_type == "image/png"


def _png_base64() -> str:
    import base64
    import io

    import PIL.Image

    buffer = io.BytesIO()
    PIL.Image.new("RGB", (16, 16), (200, 10, 10)).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_upload_image_saves_the_image_and_returns_an_id_that_a_request_can_use(engine: Engine) -> None:
    """Bug: an MCP client has no way to give the server an image, so ControlNet and inpaint cannot work."""
    mcp = build_mcp(engine)
    stored = _call(mcp, "upload_image", {"data": _png_base64(), "subdir": "refs"})
    assert stored["upload_id"].startswith("upl_") and (stored["width"], stored["height"]) == (16, 16)
    assert engine.resolve_input(stored["upload_id"]).parent == (engine.input_dir / "refs").resolve()


@pytest.mark.parametrize(
    ("data", "code"),
    [("not base64 !!", "invalid_request"), ("aGVsbG8=", "invalid_request"), ("A" * 1_400_000, "upload_too_large")],
)
def test_upload_image_refuses_bad_data(engine: Engine, data: str, code: str) -> None:
    """Bug: text that is not an image, or an upload above max_upload_mb, is saved in the input directory."""
    engine.settings.max_upload_mb = 1
    with pytest.raises(Exception, match=code):
        asyncio.run(build_mcp(engine).call_tool("upload_image", {"data": data}))
    assert not list(engine.input_dir.glob("upl_*"))


def test_a_client_can_learn_the_controlnet_workflow_from_the_mcp_alone(settings) -> None:
    """Bug: an MCP client without a skill does not know how to give an image, that it must make the control image,
    or that white in a mask is drawn again."""
    from mlx_imagine.engine import Engine as RealEngine

    mcp = build_mcp(RealEngine(settings))
    assert "upload_image" in mcp.instructions and "preview_mask" in mcp.instructions
    tools = _tools(RealEngine(settings))
    qwen = tools["generate_qwen_image_2_1_controlnet_union"]
    assert "control image" in qwen.description and "upload_image" in qwen.description
    properties = qwen.input_schema["properties"]
    assert "White is drawn again" in properties["mask_image"]["description"]
    assert "does not make the image" in properties["controlnet_image_path"]["description"]
    assert "scribble" in properties["controlnet_image_path"]["description"] and "Both" in qwen.description
    assert "upload_image" in properties["image_path"]["description"]
    zimage = tools["generate_z_image_turbo_controlnet_union_2_1"]
    assert "canny" in zimage.input_schema["properties"]["control"]["description"]
    assert "upload_image" in zimage.description
