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
- The server has no TLS (`server.use_tls` is not supported). To reach it over a network, bind it to `127.0.0.1`, and put a TLS proxy in front. [deploy/Caddyfile.example](../deploy/Caddyfile.example) is a Caddy example. It turns off the response buffer, which the event stream of a job needs.
- At start, the server logs its address and each enabled service. It logs a warning when no token is on.

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
| `option_ignored` | 422 | The model ignores an option of the request, for example `guidance` for a guidance-distilled model, or `negative_prompt` at guidance 1.0. |
| `limit_exceeded` | 422 | A generate request is above `limits.max_pixels`, `max_steps` or `max_seeds`. |
| `body_too_large` | 413 | A JSON body is larger than 1 MB, or an upload (`/v1/uploads` or `/v1/images/edits`) is larger than `max_upload_mb` plus 1 MB. |
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
  "restart_required": false,
  "permissions": {
    "hf_access": true,
    "model_download": false,
    "download_needs_confirm": true,
    "max_download_gb": 0.4,
    "hf_cache_delete": true,
    "mflux_dir_delete": true,
    "lora_delete": true
  },
  "lora_directory": "/Users/me/mlx-imagine/loras"
}
```

`restart_required` is `true` after a change to a setting that needs a restart (3.3).

`permissions` shows what the config allows, so a client does not learn about a block from a refused call. `model_download` is `huggingface_models.allow_model_download`. It also controls the LoRA download (4.6). `max_download_gb` is `huggingface_models.max_download_model_size_gb`, or `null` for no limit. `hf_cache_delete` and `mflux_dir_delete` tell if a delete from each store is allowed (4.5). `lora_directory` is the folder for LoRA files.

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
| `mflux_models.manifest_url` | string | the community manifest URL | No | The JSON Lines file with the mflux-community models. |
| `mflux_models.cron_minutes` | integer | `5` | No | The minutes between scans of the mflux model directory. `0` turns the scan off. |
| `mflux_models.allow_download` | boolean | `true` | No | `false` refuses the download of a community mflux model into `local_directory` (4.3). |
| `mflux_models.allow_delete` | boolean | `true` | No | `false` refuses each delete of a saved model. |
| `mflux_models.max_models` | integer | none | No | A save fails when the directory holds this number of models. |
| `mflux_models.max_size_gb` | number | none | No | A save fails when the models use this size. |
| `loras.local_directory` | path | `~/mlx-imagine/loras` | No | The LoRA files (4.6). |
| `loras.allow_delete` | boolean | `true` | No | `false` refuses each delete of a LoRA file. |
| `huggingface_models.local_directory` | path | `huggingface_hub` default | Yes | The Hugging Face hub cache (the folder with the `models--*` folders). Sets `HF_HUB_CACHE`. |
| `huggingface_models.cron_minutes` | integer | `5` | No | The minutes between scans of the Hugging Face cache. The manifest sync runs with it. `0` turns both off. |
| `huggingface_models.allow_hf_access` | boolean | `true` | Yes | `false` sets `HF_HUB_OFFLINE=1`. The server never reaches Hugging Face, and a download request returns `operation_disabled`. |
| `huggingface_models.allow_model_download` | boolean | `true` | No | `false` lets a download fetch small files only (JSON, text). It skips weight files (`.safetensors`, `.ckpt`, `.gguf`, `.bin`, `.pt`, `.pth`, `.onnx`, `.h5`, `.msgpack`). The job result has `weights_skipped: true`, and `complete` stays `false`. |
| `huggingface_models.require_model_confirmation` | boolean | `false` | No | `true` makes a download request need `confirm: true`. Without it, the server returns `confirm_required` and the size of the download. Applies only when `allow_model_download` is `true`. |
| `huggingface_models.allow_delete` | boolean | `true` | No | `false` refuses each delete from the cache. |
| `huggingface_models.max_models` | integer | none | No | A download of a new model fails when the cache holds this number of mflux models. |
| `huggingface_models.max_size_gb` | number | none | No | A download of a new model fails when the mflux models in the cache use this size. |
| `huggingface_models.max_download_model_size_gb` | number | none | No | The most one download can fetch: a model or a LoRA file. A larger download returns `limit_exceeded`. The server takes the size from the Hub. When the Hub gives no size, the server refuses the download. For example, `0.4` allows a small LoRA and refuses a text encoder or a transformer. |
| `limits.max_pixels` | integer | `16777216` | No | The most pixels (width times height) of one image. A generate request above it returns `limit_exceeded`. |
| `limits.max_steps` | integer | `200` | No | The most denoising steps of one image. |
| `limits.max_seeds` | integer | `16` | No | The most seeds, which means images, in one generate request. |
| `image_metadata.exif_enabled` | boolean | `true` | No | Embed the generation data in the EXIF UserComment of each generated PNG. A generate request can set `exif_enabled` for one request. |
| `image_metadata.iptc_enabled` | boolean | `true` | No | Embed the IPTC and XMP data (prompt, seed, model) in each generated PNG. A generate request can set `iptc_enabled` for one request. |
| `image_metadata.json_sidecar_enabled` | boolean | `true` | No | Write a `.metadata.json` file next to each generated PNG. A generate request can set `json_sidecar_enabled` for one request. |
| `services.api_enable` | boolean | `true` | Yes | `false` turns off the REST routes under `/v1`, except `/v1/health`. They return `api_disabled` (HTTP 404). |
| `services.mcp_enable` | boolean | `true` | Yes | `false` does not start the MCP server at `/mcp`. |
| `services.swagger_enable` | boolean | `true` | Yes | `true` serves the Swagger page at `/docs`, ReDoc at `/redoc` and the schema at `/openapi.json`. When a token is on, these paths need no token. The page has an Authorize button for it. |
| `data_directory` | path | `~/mlx-imagine/data` | No | The folder for `mlx-imagine.sqlite` and the manifest copy. |
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
| `cache_limit_gb` | number | `1` | No | Calls `mx.set_cache_limit()`. MLX keeps freed buffers in this cache for the next job. |

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

All model routes start with `/v1/models/mflux`. The name `mflux` leaves room for other model libraries later.

### 4.1 The model list

A model has a format and a location.

| Format | Meaning |
|---|---|
| `mflux` | Safetensors in the mflux layout, often quantized. |
| `hf` | Safetensors in the layout of the original Hugging Face repository. |

| Location | Store | Where |
|---|---|---|
| local | `hf_cache` | The Hugging Face cache (`huggingface_models.local_directory`). |
| local | `mflux_dir` | The mflux model directory (`mflux_models.local_directory`). |
| remote | `mflux_community` | The mflux-community models in the community manifest. |
| remote | `hub` | All of Hugging Face. A live search. |

`GET /v1/models/mflux` lists models. All filters are optional and combine with AND:

| Query | Values | Meaning |
|---|---|---|
| `location` | `local`, `remote` | The default is `local`. |
| `store` | `hf_cache`, `mflux_dir`, `mflux_community`, `hub` | One store. |
| `format` | `hf`, `mflux` | One format. |
| `supported` | `true`, `false` | Models that this server can run. |
| `family` | text | A model family, for example `flux2`. |
| `q` | text | A text in the name. The `hub` store needs it. |
| `limit`, `cursor` | integer, text | A page. The response has `next_cursor` when more models follow. The default limit is 100 (20 for a remote list). |

The views of the table in the roadmap map to requests like this:

| View | Request |
|---|---|
| Local, `hf` format | `?location=local&format=hf` |
| Local mflux format in the Hugging Face cache | `?store=hf_cache&format=mflux` |
| Local mflux format in the mflux directory | `?store=mflux_dir` |
| All local mflux format models | `?location=local&format=mflux` |
| Remote, mflux-community | `?location=remote&store=mflux_community` |
| Remote, all of Hugging Face | `?store=hub&q=<text>` |

Response:

```json
{
  "models": [
    {
      "id": "mflux_dir/z-image-turbo-q8",
      "store": "mflux_dir",
      "location": "local",
      "format": "mflux",
      "repo_id": null,
      "name": "z-image-turbo-q8",
      "model_name": "z-image-turbo",
      "base_model": "z-image-turbo",
      "family": "z-image",
      "quant": "q8",
      "size_gb": 9.1,
      "complete": null,
      "supported": true,
      "loadable": true,
      "in_manifest": false,
      "path": "/Users/me/mlx-imagine/models/z-image-turbo-q8",
      "scanned_at": "2026-10-17T10:12:00Z"
    }
  ],
  "total": 1,
  "next_cursor": null
}
```

- `id` is `<store>/<name>`. For `hf_cache`, `mflux_community` and `hub`, the name is the repository ID, so the id has a second `/`.
- `supported` is true when the server can run the base model. `loadable` is true when a load request can use this model now. A model in `hf_cache` with format `mflux` is loadable when its newest snapshot has the mflux layout. A saved model without `mlx-imagine.json` is not loadable until you set `base_model`.
- `in_manifest` is true when the community manifest names the repository.
- A `hf_cache` model has `revisions` and `last_accessed`. A `mflux_dir` model has `mflux_version`, `saved_at` and `has_sidecar`. A remote model has `url`.
- The local lists read the database, so they answer at once. They can be up to `cron_minutes` old. `POST /v1/models/mflux/rescan` scans both stores and syncs the manifest now.

`GET /v1/models/mflux/{store}/{name}` returns one model.

The database is `data/mlx-imagine.sqlite`. A thread scans the stores every `cron_minutes` and copies the manifest to `data/models_community_manifest.jsonl` when it changed. The manifest and the `hub` search reach the network only when `huggingface_models.allow_hf_access` is true. The cache scan lists a repository only when the mflux registry or the manifest names it.

### 4.2 Registry and schema

`GET /v1/models/mflux/registry` lists all models in the mflux registry:

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

#### Model schema

`GET /v1/models/mflux/schemas/{name}` returns two JSON Schemas for one model. The server makes them from the command's `build_parser()` (scope 3.6).

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

### 4.3 Download

`POST /v1/models/mflux/downloads` makes a `download` job:

```json
{ "model": "z-image-turbo", "confirm": true }
```

- `model` is a registry name, an alias or a Hugging Face repository ID in the registry. The server downloads only the files that the model needs. The files go to the Hugging Face cache.
- `model` can also be the repository ID of a quantization in the community manifest (the store `mflux_community` of 4.1), for example `mflux-community/flux2-klein-9b-kv-mflux-q4`. The server then downloads the whole repository to `mflux_models.local_directory`, in a folder named `<owner>--<name>`, and writes `mlx-imagine.json` with the base model from the manifest. The result is a saved model, so it loads by that folder name. This download needs `mflux_models.allow_download`, `huggingface_models.allow_model_download` and a base model that the server can run. The `mflux_models` limits `max_models` and `max_size_gb` apply.
- `confirm` is needed only when `huggingface_models.require_model_confirmation` is `true`.
- The job `result` is the model entry from the list.

### 4.4 Save

`POST /v1/models/mflux/saves` makes a `save` job:

```json
{ "model": "z-image-turbo", "quantize": 8, "name": "z-image-turbo-q8" }
```

- `name` is optional. The default is `<model>-q<quantize>`.
- `name` is one directory name. It cannot contain `/` or `..`.
- If the directory exists, the server returns HTTP 409.
- The save job unloads the loaded model first, because the server keeps a maximum of one model in memory. After the save, no model is loaded.
- The server writes `mlx-imagine.json` in the new directory.

`PATCH /v1/models/mflux/mflux_dir/{name}` sets `base_model` for a model that has no sidecar file:

```json
{ "base_model": "z-image-turbo" }
```

The server writes the sidecar file. It checks that `base_model` is a registry name.

### 4.5 Delete

`DELETE /v1/models/mflux/{store}/{name}?confirm=true` deletes a model from the store `hf_cache` or `mflux_dir`.

- Requires `confirm=true`. Without it, the server returns `confirm_required`.
- Returns `operation_disabled` when `allow_delete` is false for that store.
- Returns `model_in_use` when the model is loaded, is the target model, or a job in the queue uses it.
- A delete from `hf_cache` uses `huggingface_hub` `delete_revisions()`. A delete from `mflux_dir` removes only a directory inside `mflux_models.local_directory`.
- Returns HTTP 200 with the freed size.

### 4.6 LoRA files

The LoRA files are in `loras.local_directory`. A LoRA name is the path of the file inside that directory, for example `style/film.safetensors`. The server finds `.safetensors` files up to 2 folders deep and skips hidden folders.

| Method and path | Description |
|---|---|
| `GET /v1/loras` | List the LoRA files: `{"directory", "loras": [{"name", "size_mb"}]}`. `directory` is the LoRA directory. |
| `POST /v1/loras/downloads` | Download one file from a Hugging Face repository. Makes a `lora_download` job. |
| `DELETE /v1/loras/{name}?confirm=true` | Delete a LoRA file. |

`POST /v1/loras/downloads` request:

```json
{ "repo_id": "owner/name", "filename": "style.safetensors", "confirm": true }
```

- `repo_id` is `owner/name`. `filename` is one `.safetensors` file inside the repository, in the top folder or in one subfolder. It has no wildcard.
- The server saves the file as `owner--name/<filename>` in the LoRA directory. It downloads only that file. If the file exists already, the server returns `name_in_use`.
- The download follows `huggingface_models.allow_hf_access`, `allow_model_download` and `require_model_confirmation`. When `allow_model_download` is false, the answer is `operation_disabled`. It names the LoRA directory, where a client can put the file. With confirmation on, the server returns `confirm_required` and the size.
- A delete needs `confirm=true`. It returns `operation_disabled` when `loras.allow_delete` is false, and `lora_in_use` when the loaded model, the target model or a queued load uses the file.
- The server never passes a Hugging Face name or a client path to mflux. A client applies a LoRA only through a file in the LoRA directory.

To apply a LoRA, give `loras` in the load request (5.1).

### 4.7 ControlNet and inpainting

ControlNet is only available for a model that supports it. The server runs these models (the ControlNet checkpoint comes from its own repository, which a download request fetches with the base model):

| Model | Request needs | Notes |
|---|---|---|
| `qwen-image-2.1-controlnet-union`, `qwen-image-2.1-turbo-controlnet-union` | `controlnet_image_path` (canny, depth, grayscale, HED, lineart, MLSD, pose or scribble), or `image_path` with `mask_image`, or all three | Alibaba-PAI Fun ControlNet Union. One checkpoint reads all control types, so a request gives no type. With `image_path` and `mask_image` the model inpaints: white in the mask is drawn again, black stays. The server resizes the mask to the output size. With all three options the model inpaints under control. `controlnet_strength` (default `1.0`) and `output_resolution` (default `1024`) are options. The Turbo model refuses `guidance` other than `1`. |
| `z-image-turbo-controlnet-union-2.1` | `control`: one or more `type:path[:strength]` | `type` is `canny`, `depth`, `hed`, `mlsd` or `pose`. Z-Image has no inpaint mode. |

The server does not make a control image. The client makes the canny edges, the depth map or the pose image, and sends it as an upload (5.3). An MCP client uses the tool `upload_image`: it takes the image as base64 text and an optional `subdir`, and returns the same answer as `POST /v1/uploads`. It follows `image_directories.max_upload_mb`.

The MCP server tells a client about this in its `instructions` and in the description of each ControlNet generate tool. The schema of each image option says that it takes an upload ID or a path inside `input_dir`.

`POST /v1/masks/preview` (MCP: `preview_mask`) takes `{"image_path", "mask_image"}` and returns a PNG of the image with the white area of the mask drawn in red. It needs no model and makes no job. Use it to check a mask before a long generate request. The MCP tool returns a JPEG preview of the same picture, with the size limits of 6.4.

A model with a ControlNet loads more weights. On a 36 GB Mac, the Qwen model with its ControlNet used about 8 GB of swap during the load. This was true also for a quantized saved model. Close other programs first.

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
| `model` | Yes | A registry name, an alias or the name of a saved model. It can also be the repository ID of an mflux-format model in the Hugging Face cache, for example `mflux-community/flux2-klein-9b-kv-mflux-q4`. |
| `base_model` | No | Required for a saved model without a sidecar file. For a Hugging Face cache model, the community manifest gives it, so it is needed only when the manifest does not name the model. |
| `quantize` | No | The default is `default_quantize`. Not allowed for a saved model or an mflux-format model, which is already quantized. |
| `low_ram` | No | Default `false`. See below. |
| `loras` | No | LoRA files from `GET /v1/loras`: a list of names, or of `{"name", "scale"}` objects. The scale is a number and the default is `1.0`. At most 8. mflux merges them into the weights at load. To change the LoRAs, send a new load request. A model that has no LoRA support returns `lora_not_supported`. |
| other options | No | The other options in `load_schema` (4.2). |

Rules:

- If a model is loaded, the load job unloads it first.
- The job checks the free memory before it loads. If the check fails, the job fails with `insufficient_memory`. The old model is then already unloaded.
- The load job does not download. If the model is not complete in the Hugging Face cache, the load request returns `not_found` at once. Send a download request first. The message names the saved models in the mflux directory that are copies of the model. A load request for one of them needs `base_model`.

Low-RAM mode (scope 3.8):

- With `low_ram: true`, each generate job frees parts of the model.
- After each generate job, the server loads the model again before the next generate job. This load is part of the next generate job. `GET /v1/model` shows `"state": "needs_reload"` between jobs.

`GET /v1/model`:

```json
{ "name": "z-image-turbo", "quantize": 8, "low_ram": false, "state": "ready", "loaded_at": "2026-10-17T09:20:00Z" }
```

`state` is `ready` or `needs_reload`.

`capabilities` helps a client choose request values:

| Field | Meaning |
|---|---|
| `modes` | `txt2img`, `img2img`, `edit`, `controlnet` or `inpaint`. A ControlNet model has `controlnet` and, for Qwen-Image 2.1, `inpaint`. It does not have `txt2img`: each request needs a control image, or an image and a mask. |
| `default_steps` | The number of steps when the request has none. |
| `supports_guidance` | `false` when the model ignores `guidance`. A request that sets it returns `option_ignored`. |
| `default_guidance` | The guidance when the request has none, or `null` when the model decides. |
| `supports_negative_prompt` | `false` when the model never uses `negative_prompt`. At low guidance a model can ignore it too. |
| `supports_lora` | `true` when a load request can use `loras`. |

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
- `exif_enabled`, `iptc_enabled` and `json_sidecar_enabled` are optional booleans. A value replaces the `image_metadata` setting for this request. They are not mflux options, and they show in the generate schema and in the generate tools of MCP.
- `image_path` is an upload ID (5.3), or a path inside `input_dir`. A different path returns `path_not_allowed`.
- Edit models take `image_paths`, a list of reference images (each an upload ID or a path inside `input_dir`). Qwen-Image 2.1 takes 1 to 10. The prompt calls them `<image1>`, `<image2>` and so on. With `mask_image` the model repaints only the first image. `strength` below 1 keeps more of the first image.
- ControlNet models (4.7): `controlnet_image_path` and `mask_image` are image options like `image_path`: an upload ID, or a path inside `input_dir`. `control` is a list of strings `type:path[:strength]` (Z-Image). Each path in a control spec follows the same rule.
- `input_subdir` and `output_subdir` are optional strings. They are not mflux options, and they show in the generate schema and in the generate tools of MCP. A value is a relative path of 1 to 3 folder names, for example `cats/2026`. A name that is empty, absolute, hidden or has `..` returns `invalid_request` or `path_not_allowed`, before the job enters the queue.
  - `output_subdir`: the server saves the images of this request in that folder of `output_dir`, and makes the folder when needed. The `url` of each image works as usual.
  - `input_subdir`: a relative `image_path` is read from that folder of `input_dir`, which must exist (else `not_found`). A path outside the folder returns `path_not_allowed`. An upload ID works with or without it.

Blocked options: `output`, `prompt_file`, `config_from_metadata`, `stepwise_image_output_dir`, `battery_percentage_stop_limit`, `metadata`, `model_path`, and all LoRA options (`lora`, `lora_paths`, `lora_scales`, `lora_style`, `bake_lora`). Use `loras` in the load request instead (5.1).

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

`POST /v1/uploads?subdir=refs` saves the file in that folder of `input_dir` (5.2 gives the rules for the name). The `upload_id` finds the file in any folder.

### 5.4 Images

`GET /v1/images/{job_id}/{index}` returns the image file as `image/png`.

The server keeps all images on disk. The API has no list or delete for images (scope 3.13).

### 5.5 OpenAI-compatible routes

Three routes follow the OpenAI images API, so a client that speaks it can use the server. They use the same token as the other routes, and they wait for the images as OpenAI does.

| Route | Description |
|---|---|
| `GET /v1/models` | The local models that this server can run, as `{"object": "list", "data": [{"id", "object": "model", ...}]}`. |
| `POST /v1/images/generations` | JSON: `model`, `prompt`, `n`, `size` (`1024x1024`), `response_format` (`b64_json` or `url`). |
| `POST /v1/images/edits` | Multipart: `image`, `prompt`, `model`, `n`, `size`, `response_format`. |

- `model` is a registry name, an alias or a saved model. The route loads it first when it is not the loaded model. Without `model`, the route uses the loaded model.
- `n` is the number of images. The route picks `n` random seeds. A client can add `seed`, `steps`, `guidance` and `negative_prompt`. They are not OpenAI fields, and the rules of 5.2 apply to them.
- `response_format` defaults to `b64_json`. A `url` points to `GET /v1/images/{job_id}/{index}`, which needs the token like every route.
- A client that closes the request cancels the job.
- Not supported: streaming, partial images and masks.

## 6. MCP server

### 6.1 Transports

- Streamable HTTP: at `/mcp` on the same address and port as the REST API. It uses the same token.
- stdio: `mlx-imagine mcp`. The command is a thin client. It has no engine and no model. It passes each message to `/mcp` of the server that runs. If no server runs, the command stops with an error and does not start a server.

Only one server can run for each user. The server takes an exclusive lock on `~/Library/Application Support/mlx-imagine/server.lock` when it starts and writes its PID and address to that file. A second `serve` stops with an error that gives the PID. The system drops the lock when the process ends, so a crash leaves no stale lock.

### 6.2 Tools

Each tool calls the same server code as the REST endpoint.

| Tool | REST endpoint | Annotations |
|---|---|---|
| `server_status` | `GET /v1/server` | read-only |
| `get_config` | `GET /v1/config` | read-only |
| `set_config` | `PATCH /v1/config` | |
| `list_registry_models` | `GET /v1/models/mflux/registry` | read-only |
| `list_models` | `GET /v1/models/mflux` | read-only |
| `download_model` | `POST /v1/models/mflux/downloads` | |
| `save_model` | `POST /v1/models/mflux/saves` | |
| `set_saved_model_base` | `PATCH /v1/models/mflux/mflux_dir/{name}` | |
| `delete_model` | `DELETE /v1/models/mflux/{store}/{name}` | destructive |
| `upload_image` | `POST /v1/uploads` | |
| `preview_mask` | `POST /v1/masks/preview` | read-only |
| `list_loras` | `GET /v1/loras` | read-only |
| `download_lora` | `POST /v1/loras/downloads` | |
| `delete_lora` | `DELETE /v1/loras/{name}` | destructive |
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

- Default: `45` for generate tools, `0` for the other tools. A tool waits at most `55` seconds, and a larger `wait_seconds` is cut to `55`. Each MCP host ends a tool call at its own limit. Claude Code failed a call at about 60 seconds.
- The tool sends MCP progress notifications while it waits.
- If the job ends in the wait time, the tool returns the result. A generate tool also returns each image as MCP image content. This is a JPEG preview of at most 1024 pixels, because a full PNG can be too large for an MCP client. The tool lowers the size and the quality until all images of the result fit in about 700 KB of base64 text. A transparent area is drawn on white. The `path` in the job result is the full image.
- If the job does not end in the wait time, the tool returns the job object. It also returns a text that tells the agent to call `get_job` with the job ID. The job continues on the server.

## 7. Command line

These commands are not part of the API. They are listed here to show the full surface.

| Command | Description |
|---|---|
| `mlx-imagine` | Start the server. Same as `mlx-imagine serve`. |
| `mlx-imagine serve [--host] [--port] [--config]` | Start the server. The flags change the config for this run only. |
| `mlx-imagine mcp` | MCP over stdio for the server that runs. It starts no server (6.1). |
| `mlx-imagine service install [--force]` | macOS only. Install the LaunchAgent (scope 3.17). |
| `mlx-imagine service uninstall` | macOS only. Stop the server and delete the LaunchAgent. |
| `mlx-imagine service start` | macOS only. Start the server. Installs the LaunchAgent first when needed. |
| `mlx-imagine service stop` | macOS only. Stop the server. The LaunchAgent stays installed. |
| `mlx-imagine service restart` | macOS only. Restart the server. |
| `mlx-imagine service status` | macOS only. Show the LaunchAgent state and the server health. |

The REST API and the MCP server cannot install or uninstall the service.

## 8. Items to review

1. The default port `8765`.
2. The default directories under `~/mlx-imagine/`.
3. The `wait_seconds` default of 45 seconds and the maximum of 55 seconds for MCP generate tools.
4. The blocked option list in 5.2.
