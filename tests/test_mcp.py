"""The MCP tools: names, annotations, generated schemas and one generate call over the fake command."""

import asyncio
import json

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
