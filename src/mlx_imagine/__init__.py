"""mlx-imagine: a long-running HTTP and MCP server for mflux image models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlx-imagine")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"
