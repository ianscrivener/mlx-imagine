# mlx-imagine

> Local AI image creation on your Mac M-Series computer.
> It is AI-first: your AI agent drives it over MCP.
> It also has a REST API for programmatic image generation.
> It builds on [`MFlux`](https://github.com/mflux-community/mflux)




is a server built on [mflux](https://github.com/mflux-community/mflux), which it uses as a library. Its command, `mlx-imagine`, runs a server that keeps one mflux image model in memory and runs one job at a time. Clients use a REST API or an MCP server.

Status: stage 1, pre-release. The tests use fake models only. Nobody has run this version with real weights yet.

## What it does

- Load and unload an mflux model.
- Generate images from text, or from an input image when the model supports it.
- List the compatible models in the Hugging Face cache, download them and delete them.
- Save a quantized copy of a model, list the saved models and delete them.
- Read and change the server settings.
- On macOS, install the server as a LaunchAgent that starts at login.

Training is not supported. Use `mflux-train`.


---
### Install

```sh
uv tool install mflux --with-executables-from mlx-imagine
```

From a checkout:

```sh
uv sync --locked
uv run mlx-imagine --help
```

---
### Start the server

```sh
mlx-imagine
```

The server listens on `http://127.0.0.1:8765`. It starts with no model loaded.

Example session:

```sh
curl -X POST localhost:8765/v1/models/mflux/downloads -H 'content-type: application/json' -d '{"model": "z-image-turbo"}'
curl -X POST localhost:8765/v1/model/load -H 'content-type: application/json' -d '{"model": "z-image-turbo", "quantize": 8}'
curl -X POST localhost:8765/v1/generate -H 'content-type: application/json' -d '{"prompt": "a puffin on a rock", "seed": [1, 2]}'
curl localhost:8765/v1/jobs/<job id>
```

Each request that takes time returns a job. Read the job with `GET /v1/jobs/{id}`, or follow it with `GET /v1/jobs/{id}/events` (Server-Sent Events). The full API is in [docs/api.md](docs/api.md).

The request fields use the names of the mflux command options: `--image-path` becomes `image_path`. `GET /v1/models/mflux/schemas/{name}` gives the JSON Schema of each model.

## Settings

The config file is `~/.config/mlx-imagine/config.yaml`. Set `MLX_IMAGINE_CONFIG` or `--config` to use a different file.

```yaml
mflux_config:
  default_quantize: 8
  allowed_quantize: "q4 q6 q8 bf16"
mflux_models:
  local_directory: /Volumes/Models/mflux
  allow_delete: true
huggingface_models:
  local_directory: /Volumes/Models/huggingface/hub
image_directories:
  output: ./output
server:
  port: 8765
memory_limit_gb: 28
```

[docs/api.md](docs/api.md#33-configuration) lists all settings. A relative path is relative to the config file. A file with the suffix `.toml` is read as TOML. `PATCH /v1/config` changes the settings and writes the file. The write does not keep the comments of the file. A change to a directory of the Hugging Face or mflux cache, or to a `server` setting, applies after a restart.

To bind to an address other than `127.0.0.1`, set `server.require_api_token: true` and `server.api_token`. Clients then send `Authorization: Bearer <token>`.

## MCP

The server has a Streamable HTTP MCP endpoint at `http://127.0.0.1:8765/mcp`. For Claude Code:

```sh
claude mcp add --transport http mflux http://127.0.0.1:8765/mcp
```

For a client that uses stdio, run `mlx-imagine mcp`. The command stops with an error when a server already runs, because two processes would each load a model.

[docs/mcp.md](docs/mcp.md) tells how to connect a client. [docs/api.md](docs/api.md#6-mcp-server) lists the tools. Each supported model has its own generate tool, for example `generate_z_image_turbo`. A generate tool loads the model if necessary, waits for the images and returns them.

## Run at login (macOS)

```sh
mlx-imagine service install
mlx-imagine service status
mlx-imagine service start    # also: stop, restart
mlx-imagine service uninstall
```

The logs are in `~/Library/Logs/mlx-imagine/`. After a crash, `launchd` starts the server again, with 30 seconds between attempts.

## mflux versions

mlx-imagine needs mflux 0.23 or later. It calls command steps and some private mflux functions that can change in any mflux release. `tests/test_mflux_drift.py` fails when one of them changes.

## Development

| Task | Command |
|---|---|
| Install | `just venv-install` |
| Tests (no weights) | `just test` |
| All checks before a commit | `just dev-check` |
| Lint, format, types and docs in one go | `just dev` (`just dev-fix` applies the fixes) |
| Type check | `just dev-types` |
| Real-weight checks (asks first) | `just test-real` |
| mflux from git main, a PR or a branch | `just mflux-bump`, `just mflux-pr 886`, `just mflux-branch NAME` |
| Back to the locked mflux | `just mflux-reset` |

`just` lists all recipes.

`just test` does not download a model or load weights. The tests use a fake model class behind the real mflux parser and the real `validate()` step.

## Design

- [docs/scope.md](docs/scope.md): the proposal, the decisions and the scope of work.
- [docs/api.md](docs/api.md): the REST API and the MCP tools.
- [docs/mflux-internals.md](docs/mflux-internals.md): each part of mflux that mlx-imagine uses, and its risk.

## License

Apache-2.0. See [LICENSE](LICENSE). [NOTICE](NOTICE) lists the code copied from mflux (MIT).
