"""A command-line MCP client for the customer tests. It calls the MCP server of a running mlx-imagine, so an AI agent
that has no MCP connection can use the same tools, with the same names, descriptions and schemas.

  mcp_cli.py tools                  List each tool: name, description and input schema.
  mcp_cli.py call NAME 'JSON'       Call a tool. NAME is the tool name. JSON is the arguments object.

A string value that starts with @file: is replaced by the base64 text of that file. This lets an agent
that cannot write a large base64 text send an image, for example {"data": "@file:images/a.png"}.
An image in the result is saved to a file, and the output gives its path."""

import asyncio
import base64
import json
import sys
import tempfile
from pathlib import Path

URL = "http://127.0.0.1:8765/mcp"


def expand(value):
    if isinstance(value, str) and value.startswith("@file:"):
        return base64.b64encode(Path(value[6:]).expanduser().read_bytes()).decode()
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    return value


async def main() -> None:
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if sys.argv[1] == "tools":
                result = await session.list_tools()
                for tool in result.tools:
                    print(f"## {tool.name}\n{tool.description}\ninput schema: {json.dumps(tool.input_schema)}\n")
            elif sys.argv[1] == "call":
                result = await session.call_tool(sys.argv[2], expand(json.loads(sys.argv[3])))
                for item in result.content:
                    if item.type == "text":
                        print(item.text)
                    elif item.type == "image":
                        out = Path(tempfile.mkstemp(suffix=".png", prefix="mcp_image_")[1])
                        out.write_bytes(base64.b64decode(item.data))
                        print(f"[image saved to {out}]")
                if result.is_error:
                    sys.exit(1)
            else:
                sys.exit(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
