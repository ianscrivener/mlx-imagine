# MCP setup guide

This guide tells how to connect an AI agent to mlx-imagine over MCP. [api.md](api.md#6-mcp-server) lists each tool.

## 1. How it works

One server runs on your Mac. It holds the model and the GPU. Every MCP client connects to this server. No MCP client starts a server of its own.

| Connection | Use it when | Command or address |
|---|---|---|
| Streamable HTTP | The client supports HTTP MCP. This is the best choice. | `http://127.0.0.1:8765/mcp` |
| stdio client | The client supports only stdio. | `mlx-imagine mcp` |

The command `mlx-imagine mcp` is a thin client. It has no model. It passes each message to `/mcp` of the server that runs. When no server runs, it stops with an error and does not start one.

The server takes a lock when it starts: the file `~/Library/Application Support/mlx-imagine/server.lock`. A second server stops with an error that gives the PID of the first server. The system drops the lock when the server ends, also after a crash.

## 2. Start the server

1. Install the package with `just venv-install`.
2. Edit the config file. The `just` recipes use `config.yaml` in the project folder. A server that starts without `--config` reads `~/.config/mlx-imagine/config.yaml`. Make sure that `services.mcp_enable` is `true`.
3. Start the server. Use one of these:
   - **As a service (recommended).** `just mac-service install` installs the LaunchAgent and starts the server. The server starts at login and restarts after a crash. Use `just mac-service start | stop | restart | status` afterwards.
   - **By hand.** `just dev-serve` runs the server in the terminal. Ctrl-C stops it.
4. Check the server: `curl http://127.0.0.1:8765/v8/health`.

A change to `services.mcp_enable` takes effect after a restart: `just mac-service restart`.

Do not run the server by hand while the service runs. The second server stops with an error.

## 3. Connect a client

### Claude Code

```sh
claude mcp add --transport http mlx-imagine http://127.0.0.1:8765/mcp
```

When the server needs a token, add the header:

```sh
claude mcp add --transport http mlx-imagine http://127.0.0.1:8765/mcp --header "Authorization: Bearer <token>"
```

### A client that supports only stdio

Use this entry in the MCP settings of the client. Give the full path of the `mlx-imagine` command and of the config file. The command reads the address and the token from the config file:

```json
{
  "mcpServers": {
    "mlx-imagine": {
      "command": "/path/to/mlx-imagine/.venv/bin/mlx-imagine",
      "args": ["--config", "/path/to/mlx-imagine/config.yaml", "mcp"]
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
| `mlx-imagine mcp` stops with "no server runs" | Start the server (section 2). |
| `serve` stops with "A server already runs" | Use the server that runs, or stop it first: `just mac-service stop`. The error gives the PID. |
| A tool call fails after about 60 seconds | The call waited too long. Use the default `wait_seconds` and read the result with `get_job`. |
| `model_not_loaded` | Call a generate tool, or `load_model` first. |
| `insufficient_memory` | The model is too big. Use a smaller quantization or a saved quantized model. |
