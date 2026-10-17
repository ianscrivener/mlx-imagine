# Changelog

All notable changes to mlx-imagine are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `mlx-imagine`: a REST API and an MCP server for the 10 mflux commands that have command steps.
- A job queue that runs one job at a time, with progress events and cancel.
- Load, unload and generate, with low-RAM mode.
- Hugging Face cache list, download and delete.
- Saved model list, save with quantization, and delete. The server writes `mlx-imagine.json` next to each saved model.
- A config file and a settings API.
- `mlx-imagine service install | uninstall | status` for a macOS LaunchAgent.
