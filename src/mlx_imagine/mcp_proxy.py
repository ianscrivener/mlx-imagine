"""The MCP stdio client. It holds no model. It passes each message between stdio and the MCP endpoint of the
server that runs (docs/mcp.md)."""

from typing import Any

import anyio


async def run_proxy(url: str, token: str | None) -> None:
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.stdio import stdio_server
    from mcp.shared._httpx_utils import create_mcp_http_client

    headers = {"Authorization": f"Bearer {token}"} if token else None
    async with (
        create_mcp_http_client(headers=headers) as http_client,
        stdio_server() as (stdio_in, stdio_out),
        streamable_http_client(url, http_client=http_client) as (server_in, server_out),
        anyio.create_task_group() as tasks,
    ):

        async def pipe(source: Any, target: Any) -> None:
            async with target:
                async for message in source:
                    if isinstance(message, Exception):
                        continue  # a line that is not JSON-RPC: drop it
                    await target.send(message)
            tasks.cancel_scope.cancel()  # one side closed: stop the other side too

        tasks.start_soon(pipe, stdio_in, server_out)
        tasks.start_soon(pipe, server_in, stdio_out)
