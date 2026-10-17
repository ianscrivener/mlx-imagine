"""A structural test: no route and no MCP tool puts the text of a caught exception into a response (ROADMAP T-04)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "mlx_imagine"
# ApiError text is written by the server, so a handler may use it. Any other caught exception may hold paths.
ALLOWED = {"ApiError"}


@pytest.mark.parametrize("name", ["api.py", "mcp_server.py", "jobs.py"])
def test_no_handler_binds_an_exception_that_the_server_did_not_write(name: str) -> None:
    """Bug: a route writes str(exc) of an mflux or Pillow error into the response, and the client reads an absolute
    path of this Mac."""
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.name:
            kinds = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)} if node.type else set()
            if not kinds <= ALLOWED:
                bad.append(f"{name}:{node.lineno} binds {node.name!r} for {sorted(kinds)}")
    assert not bad, bad
