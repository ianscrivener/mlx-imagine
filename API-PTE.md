# mlx-imagine API in plain technical English

This file tells what each API endpoint does. It uses short sentences and one meaning for each word. For the full fields, errors and examples, read [docs/api.md](docs/api.md).

## Read this first

- The server runs at `http://127.0.0.1:8765` by default.
- `/v8` is the native API of mlx-imagine.
- `/v1` holds only the three routes that copy the OpenAI images API.
- `/mcp` is the MCP server. The MCP tools call the same functions as the `/v8` routes.
- When `server.require_api_token` is true, each request must send the header `Authorization: Bearer <token>`. `GET /v8/health` does not need the token.
- A "job" is one task in the queue. The server runs one job at a time. A route that makes a job returns HTTP 202 and the job. Read the job with `GET /v8/jobs/{job_id}` to get its state and its result.
- A "model" is a repository in the Hugging Face cache. Its ID is `<owner>/<name>`. A model that this server saved has the owner `local`.

## Model formats

The Hugging Face cache is the only model store. The server sorts each repository in the cache into one format:

| Format | Meaning |
|---|---|
| `mflux` | A model that mflux saved. It is often quantized. |
| `hf` | A model of the mflux registry in the original Hugging Face layout. |
| `other` | A repository that is not an image model, for example a language model of another program. |

An `hf` repository under 1 GB holds only small files, not the model weights.

## Server

| Endpoint | What it does |
|---|---|
| `GET /v8/health` | Tells if the server runs. It returns the server version and the mflux version. It needs no token. |
| `GET /v8/server` | Returns the server state: the loaded model, the target model, the running job, the queue length, the memory use and the permissions. |
| `GET /v8/config` | Returns all settings, in the layout of the config file. |
| `PATCH /v8/config` | Changes one or more settings and writes them to the config file. Some settings apply only after a restart. The response names them in `restart_required`. |

## Local models (the Hugging Face cache)

The list routes read a database. A thread scans the cache every `huggingface_models.cron_minutes`, so a list can be some minutes old.

| Endpoint | What it does |
|---|---|
| `GET /v8/models` | Lists the image models: format `mflux`, and format `hf` of 1 GB or more. |
| `GET /v8/models/mflux` | Lists the `mflux` models. |
| `GET /v8/models/hf` | Lists the `hf` models of 1 GB or more. |
| `GET /v8/models/hf/all` | Lists the `hf` models of all sizes, also those under 1 GB. |
| `GET /v8/models/other` | Lists the `other` repositories. |
| `GET /v8/models/all` | Lists the `mflux`, `hf` (1 GB or more) and `other` repositories. |
| `POST /v8/models/scans` | Scans the cache and syncs the community catalog now. It does not wait for the next scheduled scan. |
| `GET /v8/models/{owner}/{name}` | Returns one model. If the last scan did not find it, the server scans the cache one more time. |
| `PATCH /v8/models/{owner}/{name}` | Records the base model of an `mflux` model. Send `{"base_model": "<registry name>"}`. The server writes the file `mlx-imagine.json` in the model folder. After this, the model can load. |
| `DELETE /v8/models/{owner}/{name}?confirm=true` | Deletes an image model from the cache. It refuses an `other` repository and a model that is in use. When the model files are symlinks to a folder outside the cache, it deletes only the links. |

The list routes accept these filters: `supported`, `family`, `q` (a text in the ID), `limit` and `cursor`.

## Community models (on the Hub)

| Endpoint | What it does |
|---|---|
| `GET /v8/models-community` | Lists the mflux-community models that you can download. The data comes from two catalog files, `models_hf.json` and `models_mflux.json`. The server copies these files on a schedule. It does not search the Hub. |

Filters:

- `q` or `quant`: one quantization, for example `4` or `bf16`.
- `ns` or `namespace`: the owner of the upstream model, for example `Qwen`, `krea` or `bfl`.
- `search` or `s`: a text anywhere in the ID. `s=qwen` finds each ID that contains `qwen`.

Each result tells if this server can run the model (`supported`) and if the cache has it already (`in_cache`).

## Registry

The registry is the list of models that mflux knows.

| Endpoint | What it does |
|---|---|
| `GET /v8/registry` | Lists each model of the mflux registry. For each model, it tells if this server can run it, and why not. |
| `GET /v8/registry/{name}/schema` | Returns the options of one model as two JSON Schemas: one for the load request and one for the generate request. |

## Download and save

| Endpoint | What it does |
|---|---|
| `POST /v8/downloads` | Downloads a model or a LoRA file into the server. It makes a job. For a model, send `{"model": "<name or repo ID>"}`. For a LoRA file, send `{"kind": "lora", "repo_id": "...", "filename": "..."}`. When the config requires it, add `"confirm": true`. |
| `POST /v8/saves` | Makes a quantized copy of a registry model. It writes the copy into the cache as `local/<name>`. It makes a job. The job unloads the loaded model first. |

## LoRA files

| Endpoint | What it does |
|---|---|
| `GET /v8/loras` | Lists the LoRA files in the LoRA directory, and gives the directory path. |
| `DELETE /v8/loras/{name}?confirm=true` | Deletes one LoRA file. It refuses a file that a loaded or queued model uses. |

To apply a LoRA, give `loras` in the load request.

## The loaded model

The server keeps a maximum of one model in memory.

| Endpoint | What it does |
|---|---|
| `GET /v8/engine/model` | Returns the loaded model and what it supports. It returns 404 when the server has no model in memory. |
| `PUT /v8/engine/model` | Loads a model. Send `{"model": "<name or repo ID>"}` and the load options. It makes a job. It does not download. If the cache does not have the full model, it returns an error at once. |
| `DELETE /v8/engine/model` | Unloads the model and frees its memory. It makes a job. |

## Generate images

| Endpoint | What it does |
|---|---|
| `POST /v8/generate` | Makes images with the loaded model. Send a `prompt` and a list of seeds. Each seed makes one image. It makes a job. |
| `POST /v8/uploads` | Stores an input image (PNG, JPEG or WebP) and returns an upload ID. Use the ID in a generate request, for example as `image_path`. |
| `POST /v8/masks/preview` | Draws a mask on an image and returns a PNG. It uses no model. Use it to check a mask before you inpaint. |
| `GET /v8/images/{job_id}/{index}` | Returns one generated image as a PNG file. |

## Jobs

| Endpoint | What it does |
|---|---|
| `GET /v8/jobs` | Lists the jobs. The filter `state` selects one state, for example `running`. |
| `GET /v8/jobs/{job_id}` | Returns one job: its state, its progress, its result or its error. |
| `DELETE /v8/jobs/{job_id}` | Cancels a job that waits or runs. |
| `GET /v8/jobs/{job_id}/events` | Sends the events of one job as a Server-Sent Events stream. The stream ends when the job ends. |

## OpenAI-compatible routes

These routes let a client that uses the OpenAI images API use this server. They wait for the images. They do not return a job.

| Endpoint | What it does |
|---|---|
| `GET /v1/models` | Lists the image models that this server can run, in the OpenAI format. |
| `POST /v1/images/generations` | Makes images from a prompt. If `model` is not the loaded model, the server loads it first. |
| `POST /v1/images/edits` | Makes images from an uploaded image and a prompt. |
