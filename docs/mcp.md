# MCP setup guide

This guide tells how to connect an AI agent to mlx-imagine over MCP. [api.md](api.md#6-mcp-server) lists each tool.

## 1. Choose a transport

| Transport | Use it when | Command or address |
|---|---|---|
| Streamable HTTP | The server runs already (for example as the LaunchAgent). This is the best choice. | `http://127.0.0.1:8765/mcp` |
| stdio | The client starts its own server process, and no other server runs. | `mlx-imagine mcp` |

Use only one server. Two processes would each load a model and use the memory twice. The command `mlx-imagine mcp` stops with an error when a server already runs, and it gives the `/mcp` URL.

## 2. Start the server

1. Install the package with `just venv-install`.
2. Edit the config file. The `just` recipes use `config.yaml` in the project folder. A server that starts without `--config` reads `~/.config/mlx-imagine/config.yaml`. Make sure that `services.mcp_enable` is `true`.
3. Start the server. Use one of these:
   - `just serve-api` runs the server in the terminal.
   - `just mac-service start` runs it as a LaunchAgent. It starts at login.
4. Check the server: `curl http://127.0.0.1:8765/v1/health`.

A change to `services.mcp_enable` takes effect after a restart: `just mac-service restart`.

## 3. Connect a client

### Claude Code

```sh
claude mcp add --transport http mflux http://127.0.0.1:8765/mcp
```

When the server needs a token, add the header:

```sh
claude mcp add --transport http mlx-imagine http://127.0.0.1:8765/mcp --header "Authorization: Bearer <token>"
```

### A client that uses stdio

Use this entry in the MCP settings of the client. Give the full path of the `mlx-imagine` command and of the config file:

```json
{
  "mcpServers": {
    "mlx-imagine": {
      "command": "/path/to/mlx-imagine/.venv/bin/mlx-imagine",
      "args": ["--config", "/Users/ianscrivener/MFlux/mlx-imagine/config.yaml", "mcp"]
    }
  }
}
```

### A client that supports only stdio, with a server that runs

Use a stdio-to-HTTP bridge such as `mcp-remote`:

```json
{
  "mcpServers": {
    "mflux": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8765/mcp"]
    }
  }
}
```

## 4. The token

The server checks a token only when `server.require_api_token` is `true`. The same token protects `/mcp` and the REST API. The client sends it in the header `Authorization: Bearer <token>`. The server binds to `127.0.0.1` by default. A bind to another address needs a token.

## 5. Use the tools

An agent makes an image like this:

1. `list_registry_models` shows the models that this server can run (`supported: true`).
2. `list_models` shows the models on disk. Use the filters `location`, `format` and `store` to narrow the list.
3. `generate_<model>`, for example `generate_z_image_turbo`, loads the model when it is not loaded. It then waits and returns the images.

Rules for long jobs:

- A generate tool waits 45 seconds by default and never more than 55 seconds. Each MCP host ends a call at its own limit.
- When the job is not done, the tool returns the job ID. The agent then calls `get_job` with that ID. The job continues on the server.
- `download_model` can answer `confirm_required` with the size of the download. The agent must ask you first. It then calls the tool again with `confirm: true`.
- A delete tool needs `confirm: true`.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The client cannot connect | The server does not run, or `services.mcp_enable` is `false`. Run `just mac-service status`. |
| `401 unauthorized` | The token is missing or wrong. Send `Authorization: Bearer <token>`. |
| `mlx-imagine mcp` stops with "a server runs" | Connect the client to `/mcp` instead (section 1). |
| A tool call fails after about 60 seconds | The call waited too long. Use the default `wait_seconds` and read the result with `get_job`. |
| `model_not_loaded` | Call a generate tool, or `load_model` first. |
| `insufficient_memory` | The model is too big. Use a smaller quantization or a saved quantized model. |
