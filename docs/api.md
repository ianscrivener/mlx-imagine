# `mlx-imagine` API Specification

Status: draft for review. Version: `v1`. Stage: 1.

This document specifies the REST API and the MCP tools of `mlx-imagine`. The design decisions come from [scope.md](scope.md). Section numbers such as "scope 3.6" refer to that document.

FastAPI makes the OpenAPI file from the code. This document is the reference for review before the code exists.

## 1. Conventions

### 1.1 Address and version

- Default address: `http://127.0.0.1:8765`.
- All paths start with `/v1`.
- Requests and responses use JSON, unless this document states a different type.

### 1.2 Authentication

- Header: `Authorization: Bearer <token>`.
- On `127.0.0.1`, the token is optional. If the config sets a token, the server requires it.
- On a different address, the server requires a token. The server does not start without one.
- `GET /v1/health` does not require a token.

### 1.3 Errors

Each error response has this body:

```json
{
  "error": {
    "code": "model_not_loaded",
    "message": "No model is loaded. Send POST /v1/model/load first.",
    "details": {}
  }
}
```

| HTTP status | Use |
|---|---|
| 400 | The request is not valid JSON, or a field has the wrong type. |
| 401 | The token is missing or not correct. |
| 403 | A path is outside the configured directories. |
| 404 | The model, job, image or upload does not exist. |
| 409 | The request conflicts with the server state. Example: delete of the loaded model. |
| 413 | An upload is larger than the limit. |
| 422 | The mflux option check failed. `details` names the option. |
| 429 | The queue is full. |
| 501 | The operation is not supported on this platform or in this stage. |

Error codes:

| Code | HTTP status | Meaning |
|---|---|---|
| `invalid_request` | 400 | The body is not valid. |
| `unauthorized` | 401 | The token is missing or not correct. |
| `path_not_allowed` | 403 | A path is outside the configured directories. |
| `not_found` | 404 | The item does not exist. |
| `model_not_loaded` | 409 | A generate job has no model to use. |
| `model_in_use` | 409 | The model is loaded, or a job in the queue uses it. |
| `confirm_required` | 409 | A delete request does not have `confirm` set to `true`. A download request does not have it either, and `require_model_confirmation` is `true`. |
| `option_invalid` | 422 | An mflux option has a value that is not correct. |
| `option_blocked` | 422 | The server does not accept this mflux option (scope 3.6). |
| `option_for_load` | 422 | This option belongs to the load request, not to the generate request. |
| `model_not_supported` | 422 | The model has no command steps, or it is in stage 2. |
| `base_model_required` | 422 | A saved model has no sidecar file. Give `base_model`. |
| `queue_full` | 429 | The queue has the maximum number of jobs. |
| `operation_disabled` | 403 | The config does not allow this operation. |
| `limit_reached` | 409 | A model limit of the config stops the request. |
| `not_supported` | 501 | The operation is not supported here. |

A job that fails has the same error object in its `error` field. These codes occur only in jobs:

| Code | Meaning |
|---|---|
| `insufficient_memory` | The memory check before load failed. |
| `download_failed` | The Hugging Face download failed. |
| `generation_failed` | mflux raised an error during generation. |
| `cancelled` | A client cancelled the job. |

## 2. Jobs

### 2.1 Job model

All operations that can take a long time are jobs. These are load, unload, generate, download, and save and quantize. The server runs one job at a time. The other jobs wait in a queue, in the order they arrive (scope 3.7).

A request that makes a job returns HTTP 202 with the job object:

```json
{
  "id": "job_01J9Z3K8Q4",
  "type": "generate",
  "state": "queued",
  "position": 2,
  "created_at": "2026-10-17T09:30:00Z",
  "started_at": null,
  "finished_at": null,
  "progress": null,
  "request": { "prompt": "a cat", "seed": [1, 2, 3], "steps": 8 },
  "result": null,
  "error": null
}
```

| Field | Description |
|---|---|
| `type` | `load`, `unload`, `generate`, `download` or `save`. |
| `state` | `queued`, `running`, `succeeded`, `failed` or `cancelled`. |
| `position` | The place in the queue. `0` means that the job runs now. `null` after the job ends. |
| `progress` | See 2.2. |
| `result` | Set when `state` is `succeeded`. The type of the job sets its form. |
| `error` | Set when `state` is `failed` or `cancelled`. |

### 2.2 Progress

Generate job:

```json
{ "image_index": 1, "image_count": 3, "step": 4, "total_steps": 8 }
```

Download job:

```json
{ "bytes_done": 1200000000, "bytes_total": 12400000000, "file": "transformer/0.safetensors" }
```

Load, unload and save jobs:

```json
{ "phase": "loading_transformer" }
```

### 2.3 The target model

The server checks a generate request when the request arrives, not when the job runs. It checks the request against the **target model**. The target model is the model that is loaded after all load and unload jobs in the queue have run.

Example: the queue holds a load job for `z-image-turbo`. A generate request that arrives after it is checked against `z-image-turbo`.

### 2.4 Job endpoints

| Method and path | Description |
|---|---|
| `GET /v1/jobs` | List the jobs that wait, the job that runs and the last jobs that ended. |
| `GET /v1/jobs/{id}` | Get one job. |
| `GET /v1/jobs/{id}/events` | Server-Sent Events stream for one job. |
| `DELETE /v1/jobs/{id}` | Cancel a job. |

`GET /v1/jobs` query parameters:

| Parameter | Default | Description |
|---|---|---|
| `state` | all | Show only jobs with this state. |
| `limit` | `50` | The maximum number of jobs that ended to show. |

`GET /v1/jobs/{id}/events` sends these events:

| Event | Data |
|---|---|
| `state` | The job object, each time `state` changes. |
| `progress` | The `progress` object. |
| `image` | One item of the generate result (see 5.2), when each image is complete. |
| `end` | The final job object. The server then closes the stream. |

`DELETE /v1/jobs/{id}`:

- A job that waits leaves the queue at once. Its state becomes `cancelled`.
- A job that runs stops at the next step. Its state becomes `cancelled`.
- For a generate job with many seeds, the images that are complete stay on disk and stay in `result`.
- A job that ended returns HTTP 409.

## 3. Server and configuration

### 3.1 Health

`GET /v1/health` returns HTTP 200:

```json
{ "status": "ok", "server_version": "0.1.0", "mflux_version": "0.23.0" }
```

### 3.2 Server state

`GET /v1/server`:

```json
{
  "platform": "darwin-arm64",
  "loaded_model": { "name": "z-image-turbo", "quantize": 8, "low_ram": false },
  "target_model": { "name": "z-image-turbo", "quantize": 8, "low_ram": false },
  "running_job": "job_01J9Z3K8Q4",
  "queue_length": 2,
  "memory": {
    "system_total_gb": 36.0,
    "mlx_active_gb": 11.2,
    "mlx_peak_gb": 14.9,
    "mlx_cache_gb": 0.8,
    "mlx_memory_limit_gb": 30.0,
    "mlx_cache_limit_gb": 2.0
  },
  "restart_required": false
}
```

`restart_required` is `true` after a change to a setting that needs a restart (3.3).

### 3.3 Configuration

Two things describe a model: its format and its location.

| Format | Meaning | Settings section |
|---|---|---|
| mflux format | Safetensors in the layout of an mflux saved model. Often quantized. | `mflux_models` |
| Hugging Face format | Safetensors in the layout of the original repository, in the Hugging Face hub cache. | `huggingface_models` |

| Location | Meaning | Setting |
|---|---|---|
| Local | A folder on this Mac. | `local_directory` |
| Remote | A Hugging Face address that the server can download from. | `remote_url` (mflux format only) |

Each section has a `local_directory`. `mflux_models` also has a `remote_url`: the Hugging Face organization with ready-made mflux models. `huggingface_models` needs no URL, because its remote location is the Hugging Face Hub.

`GET /v1/config` returns all settings in the layout of the config file. `PATCH /v1/config` changes one or more settings and writes them to the config file. The default file is `~/.config/mlx-imagine/config.yaml`. The write keeps the keys that the server does not use yet, but not the comments.

| Setting | Type | Default | Restart | Description |
|---|---|---|---|---|
| `mflux_config.default_quantize` | integer | none | No | The quantization for a load request without `quantize`. It must be in `allowed_quantize`. |
| `mflux_config.allowed_quantize` | string | `q3 q4 q5 q6 q8 bf16` | No | The quantizations that load and save requests can use. `bf16` allows a load without quantization. |
| `mflux_config.cache_dir` | path | mflux default | Yes | The mflux cache. Sets `MFLUX_CACHE_DIR`. |
| `mflux_models.local_directory` | path | `~/mlx-imagine/models` | No | The saved mflux models. |
| `mflux_models.remote_url` | string | `https://huggingface.co/mflux-community` | No | The Hugging Face organization with ready-made mflux models. Not used yet. For a later download of saved mflux models. |
| `mflux_models.allow_download` | boolean | `true` | No | Not used yet. For a later download of saved mflux models. |
| `mflux_models.allow_delete` | boolean | `true` | No | `false` refuses each delete of a saved model. |
| `mflux_models.max_models` | integer | none | No | A save fails when the directory holds this number of models. |
| `mflux_models.max_size_gb` | number | none | No | A save fails when the models use this size. |
| `huggingface_models.local_directory` | path | `huggingface_hub` default | Yes | The Hugging Face hub cache (the folder with the `models--*` folders). Sets `HF_HUB_CACHE`. |
| `huggingface_models.allow_hf_access` | boolean | `true` | Yes | `false` sets `HF_HUB_OFFLINE=1`. The server never reaches Hugging Face, and a download request returns `operation_disabled`. |
| `huggingface_models.allow_model_download` | boolean | `true` | No | `false` lets a download fetch small files only (JSON, text). It skips weight files (`.safetensors`, `.ckpt`, `.gguf`, `.bin`, `.pt`, `.pth`, `.onnx`, `.h5`, `.msgpack`). The job result has `weights_skipped: true`, and `complete` stays `false`. |
| `huggingface_models.require_model_confirmation` | boolean | `false` | No | `true` makes a download request need `confirm: true`. Without it, the server returns `confirm_required` and the size of the download. Applies only when `allow_model_download` is `true`. |
| `huggingface_models.allow_delete` | boolean | `true` | No | `false` refuses each delete from the cache. |
| `huggingface_models.max_models` | integer | none | No | A download of a new model fails when the cache holds this number of mflux models. |
| `huggingface_models.max_size_gb` | number | none | No | A download of a new model fails when the mflux models in the cache use this size. |
| `image_directories.input` | path | `~/mlx-imagine/input` | No | Uploads and input images. |
| `image_directories.output` | path | `~/mlx-imagine/output` | No | Generated images. |
| `image_directories.max_upload_mb` | integer | `50` | No | The maximum size of one upload. |
| `server.host` | string | `127.0.0.1` | Yes | The address the server binds to. A different address requires a token. |
| `server.port` | integer | `8765` | Yes | The port. |
| `server.require_api_token` | boolean | `false` | Yes | `true` makes clients send `api_token`. |
| `server.api_token` | string | none | Yes | The token. `GET` shows `"***"` only. |
| `server.use_tls` | boolean | `false` | Yes | Not supported yet. With `true`, the server does not start. |
| `queue.queue_max_length` | integer | `100` | No | The maximum number of jobs that wait. |
| `queue.job_history` | integer | `200` | No | The number of ended jobs that the server keeps in memory. |
| `memory_limit_gb` | number | none | No | Calls `mx.set_memory_limit()`. |
| `cache_limit_gb` | number | `4` | No | Calls `mx.set_cache_limit()`. MLX keeps freed buffers in this cache for the next job. |

A relative path in the config file is relative to the directory of the file. The server logs a key that it does not use, and starts. A `PATCH` request with an unknown key fails with `invalid_request`.

`PATCH /v1/config` request:

```json
{ "memory_limit_gb": 28, "huggingface_models": { "local_directory": "/Volumes/Extreme2Tb/hub" } }
```

Response:

```json
{
  "config": { "...": "all settings" },
  "applied": ["memory_limit_gb"],
  "restart_required": ["huggingface_models.local_directory"]
}
```

The server checks each path setting. A path must be a directory that exists, or a directory that the server can make.

A request that the config does not allow fails with `operation_disabled` (HTTP 403). A save or download that would go past `max_models` or `max_size_gb` fails with `limit_reached` (HTTP 409).

## 4. Models

### 4.1 Supported models

`GET /v1/models/supported` lists all models in the mflux registry:

```json
{
  "models": [
    {
      "name": "z-image-turbo",
      "aliases": ["zit"],
      "family": "z_image",
      "command": "mflux-generate-z-image-turbo",
      "hf_repo": "Tongyi-MAI/Z-Image-Turbo",
      "supported": true,
      "modes": ["txt2img", "img2img"],
      "reason": null
    },
    {
      "name": "dev",
      "aliases": ["flux-dev"],
      "family": "flux",
      "command": "mflux-generate",
      "hf_repo": "black-forest-labs/FLUX.1-dev",
      "supported": false,
      "modes": [],
      "reason": "The mflux command has no command steps."
    }
  ]
}
```

`reason` is set when `supported` is `false`. The two reasons are "no command steps" and "stage 2".

### 4.2 Model schema

`GET /v1/models/supported/{name}/schema` returns two JSON Schemas for one model. The server makes them from the command's `build_parser()` (scope 3.6).

```json
{
  "name": "z-image-turbo",
  "load_schema": { "type": "object", "properties": { "quantize": { "type": "integer", "enum": [3, 4, 5, 6, 8] } } },
  "generate_schema": {
    "type": "object",
    "required": ["prompt"],
    "properties": {
      "prompt": { "type": "string" },
      "seed": { "type": "array", "items": { "type": "integer" } },
      "steps": { "type": "integer" },
      "width": { "type": "integer" },
      "height": { "type": "integer" },
      "image_path": { "type": "string" },
      "image_strength": { "type": "number" }
    }
  }
}
```

The schemas do not include blocked options. The names are the `argparse` destination names. Example: `--image-path` becomes `image_path`.

### 4.3 Hugging Face models

| Method and path | Description |
|---|---|
| `GET /v1/models/huggingface` | List the compatible models in the Hugging Face cache. |
| `POST /v1/models/huggingface/download` | Download a model. Makes a `download` job. |
| `DELETE /v1/models/huggingface/{owner}/{repo}?confirm=true` | Delete a model from the cache. |

`GET /v1/models/huggingface` shows only repositories that are in the mflux registry:

```json
{
  "models": [
    {
      "repo_id": "Tongyi-MAI/Z-Image-Turbo",
      "model_name": "z-image-turbo",
      "supported": true,
      "size_gb": 32.9,
      "revisions": ["5f0a7c3"],
      "complete": true,
      "last_accessed": "2026-10-16T21:04:00Z"
    }
  ]
}
```

`complete` is `false` when files that the model needs are missing.

`POST /v1/models/huggingface/download` request:

```json
{ "model": "z-image-turbo", "confirm": true }
```

- `model` is a registry name, an alias or a Hugging Face repository ID in the registry.
- `confirm` is needed only when `huggingface_models.require_model_confirmation` is `true`.
- The server downloads only the files that the model needs.
- The job `result` is the model entry from `GET /v1/models/huggingface`.

`DELETE`:

- Requires `confirm=true`. Without it, the server returns `confirm_required`.
- Returns `model_in_use` when the model is loaded, is the target model, or a job in the queue uses it.
- Uses `huggingface_hub` `delete_revisions()`. Returns HTTP 200 with the freed size.

### 4.4 Saved mflux models

| Method and path | Description |
|---|---|
| `GET /v1/models/saved` | List the models in `model_dir`. |
| `POST /v1/models/saved` | Save and quantize a model. Makes a `save` job. |
| `PATCH /v1/models/saved/{name}` | Set `base_model` for a model that has no sidecar file. |
| `DELETE /v1/models/saved/{name}?confirm=true` | Delete a saved model. |

`GET /v1/models/saved`:

```json
{
  "models": [
    {
      "name": "z-image-turbo-q8",
      "path": "/Users/me/mlx-imagine/models/z-image-turbo-q8",
      "model_name": "z-image-turbo",
      "base_model": "z-image-turbo",
      "quantize": 8,
      "mflux_version": "0.23.0",
      "saved_at": "2026-10-15T10:12:00Z",
      "size_gb": 9.1,
      "has_sidecar": true
    }
  ]
}
```

A model that `mflux-save` wrote has no sidecar file. For that model, `model_name`, `base_model` and `saved_at` are `null`, and `has_sidecar` is `false` (scope 3.10).

`POST /v1/models/saved` request:

```json
{ "model": "z-image-turbo", "quantize": 8, "name": "z-image-turbo-q8" }
```

- `name` is optional. The default is `<model>-q<quantize>`.
- `name` is one directory name. It cannot contain `/` or `..`.
- If the directory exists, the server returns HTTP 409.
- The save job unloads the loaded model first, because the server keeps a maximum of one model in memory. After the save, no model is loaded.
- The server writes `mlx-imagine.json` in the new directory.

`PATCH /v1/models/saved/{name}` request:

```json
{ "base_model": "z-image-turbo" }
```

The server writes the sidecar file. It checks that `base_model` is a registry name.

`DELETE` follows the same rules as 4.3. The server deletes only a directory inside `model_dir`.

## 5. Loaded model and generation

### 5.1 Load and unload

| Method and path | Description |
|---|---|
| `GET /v1/model` | Get the loaded model, or `null`. |
| `POST /v1/model/load` | Load a model. Makes a `load` job. |
| `POST /v1/model/unload` | Unload the model. Makes an `unload` job. |

`POST /v1/model/load` request:

```json
{ "model": "z-image-turbo", "quantize": 8, "low_ram": false }
```

| Field | Required | Description |
|---|---|---|
| `model` | Yes | A registry name, an alias, or the name of a saved model. |
| `base_model` | No | Required only for a saved model without a sidecar file. |
| `quantize` | No | The default is `default_quantize`. Not allowed for a saved model, which is already quantized. |
| `low_ram` | No | Default `false`. See below. |
| other options | No | The other options in `load_schema` (4.2). |

Rules:

- If a model is loaded, the load job unloads it first.
- The job checks the free memory before it loads. If the check fails, the job fails with `insufficient_memory`. The old model is then already unloaded.
- The load job does not download. If the model is not in the Hugging Face cache, the job fails with `not_found`. Send a download request first.

Low-RAM mode (scope 3.8):

- With `low_ram: true`, each generate job frees parts of the model.
- After each generate job, the server loads the model again before the next generate job. This load is part of the next generate job. `GET /v1/model` shows `"state": "needs_reload"` between jobs.

`GET /v1/model`:

```json
{ "name": "z-image-turbo", "quantize": 8, "low_ram": false, "state": "ready", "loaded_at": "2026-10-17T09:20:00Z" }
```

`state` is `ready` or `needs_reload`.

### 5.2 Generate

`POST /v1/generate` makes a `generate` job. The body uses the names in `generate_schema` (4.2).

```json
{
  "prompt": "a cat on a red chair",
  "seed": [1, 2, 3],
  "steps": 8,
  "width": 1024,
  "height": 1024
}
```

Rules:

- The server checks the body against the target model (2.3). With no target model, it returns `model_not_loaded`.
- The server builds an `argparse.Namespace` from the body with the command's `build_parser()`. It catches `SystemExit` and returns `option_invalid`.
- A blocked option returns `option_blocked`. A load option returns `option_for_load`.
- `seed` is a list. One job makes one image for each seed. If `seed` is missing, the server picks one random seed.
- `image_path` is an upload ID (5.3), or a path inside `input_dir`. A different path returns `path_not_allowed`.

Blocked options in stage 1: `output`, `prompt_file`, `config_from_metadata`, `stepwise_image_output_dir`, `battery_percentage_stop_limit`, `metadata`, `model_path`, and all LoRA options.

Job `result`:

```json
{
  "images": [
    {
      "index": 0,
      "seed": 1,
      "path": "/Users/me/mlx-imagine/output/job_01J9Z3K8Q4_1.png",
      "url": "/v1/images/job_01J9Z3K8Q4/0",
      "width": 1024,
      "height": 1024,
      "duration_seconds": 41.8
    }
  ]
}
```

The server saves each image in `output_dir` with the mflux metadata. It always writes the JSON metadata file next to the image.

### 5.3 Uploads

`POST /v1/uploads` takes `multipart/form-data` with one field, `file`. Allowed types: PNG, JPEG and WebP.

Response, HTTP 201:

```json
{ "upload_id": "upl_01J9Z4", "path": "/Users/me/mlx-imagine/input/upl_01J9Z4.png", "width": 1536, "height": 1024 }
```

A generate request can use the `upload_id` or the `path` as `image_path`.

### 5.4 Images

`GET /v1/images/{job_id}/{index}` returns the image file as `image/png`.

The server keeps all images on disk. The API has no list or delete for images (scope 3.13).

## 6. MCP server

### 6.1 Transports

- stdio: `mlx-imagine mcp`. The command starts an MCP server with its own engine in the same process. If a server already runs at the configured address, the command stops with an error that gives the `/mcp` URL. Two processes would each load a model.
- Streamable HTTP: at `/mcp` on the same address and port as the REST API. It uses the same token.

### 6.2 Tools

Each tool calls the same server code as the REST endpoint.

| Tool | REST endpoint | Annotations |
|---|---|---|
| `server_status` | `GET /v1/server` | read-only |
| `get_config` | `GET /v1/config` | read-only |
| `set_config` | `PATCH /v1/config` | |
| `list_supported_models` | `GET /v1/models/supported` | read-only |
| `list_huggingface_models` | `GET /v1/models/huggingface` | read-only |
| `download_model` | `POST /v1/models/huggingface/download` | |
| `delete_huggingface_model` | `DELETE /v1/models/huggingface/...` | destructive |
| `list_saved_models` | `GET /v1/models/saved` | read-only |
| `save_model` | `POST /v1/models/saved` | |
| `set_saved_model_base` | `PATCH /v1/models/saved/{name}` | |
| `delete_saved_model` | `DELETE /v1/models/saved/{name}` | destructive |
| `load_model` | `POST /v1/model/load` | |
| `unload_model` | `POST /v1/model/unload` | |
| `generate_<model>` | `POST /v1/generate` | |
| `list_jobs` | `GET /v1/jobs` | read-only |
| `get_job` | `GET /v1/jobs/{id}` | read-only |
| `cancel_job` | `DELETE /v1/jobs/{id}` | destructive |

The delete tools have a required `confirm` argument. It must be `true`.

### 6.3 Generate tools

- The server has one generate tool for each supported model. Example: `generate_z_image_turbo`.
- The `inputSchema` of each tool is the `generate_schema` of its model (4.2).
- If the model is not loaded, the tool makes a load job and then the generate job.
- Upload is not available through MCP. An image-to-image tool call gives `image_path` as an upload ID or as a path inside `input_dir`. Base64 data URLs are not in stage 1.

### 6.4 Wait or return

Agents work better with one call that returns the images. The tools that make a job therefore have an optional `wait_seconds` argument.

- Default: `300` for generate tools, `0` for the other tools.
- The tool sends MCP progress notifications while it waits.
- If the job ends in the wait time, the tool returns the result. A generate tool also returns each image as MCP image content.
- If the job does not end in the wait time, the tool returns the job object. The agent then calls `get_job`.

## 7. Command line

These commands are not part of the API. They are listed here to show the full surface.

| Command | Description |
|---|---|
| `mlx-imagine` | Start the server. Same as `mlx-imagine serve`. |
| `mlx-imagine serve [--host] [--port] [--config]` | Start the server. The flags change the config for this run only. |
| `mlx-imagine mcp` | Start the MCP server on stdio (6.1). |
| `mlx-imagine service install [--force]` | macOS only. Install the LaunchAgent (scope 3.17). |
| `mlx-imagine service uninstall` | macOS only. Stop the server and delete the LaunchAgent. |
| `mlx-imagine service status` | macOS only. Show the LaunchAgent state and the server health. |

The REST API and the MCP server cannot install or uninstall the service.

## 8. Items to review

1. The default port `8765`.
2. The default directories under `~/mlx-imagine/`.
3. The `wait_seconds` default of 300 seconds for MCP generate tools.
4. The blocked option list in 5.2.
