---
name: imagine-api
description: Use a running mlx-imagine server to make images. Covers the models, the options of each model, and the REST and MCP calls. Use when the user asks to make an image with mlx-imagine, or asks what the server can do.
---

# Use the mlx-imagine server

## When to use

- The user asks to make an image with mlx-imagine.
- The user asks which models or options the server supports.
- The user asks to download, save, quantize or delete a model through the server.

## Before you start

1. Make sure that a server runs: `just serve-status`, or `curl -s http://127.0.0.1:8765/v1/health`.
2. If no server runs, ask the user before you start one. A load uses much memory. Use the `imagine-testing` skill rules for memory.
3. If an MCP client has the `mlx-imagine` tools, use them. Each tool calls the same code as its REST route.

## Find the model and its options

1. List the models: `GET /v1/models/supported` (MCP: `list_supported_models`). Use only a model with `"supported": true`. For other models, tell the user the `reason`.
2. Get the options of the model: `GET /v1/models/supported/<name>/schema`.
   - `load_schema` has the options of the load request: `quantize`, `low_ram`, `base_model`.
   - `generate_schema` has the options of the generate request.
3. The option names are the mflux command-line flags with `_` for `-`. For example, `--image-path` is `image_path`.
4. Do not send `steps` or `guidance` unless the user asks for a value. The model then uses its own default. A turbo model needs few steps, and a base model needs many.

## Make an image

1. Load the model: `POST /v1/model/load` with `{"model": "<name or saved model>", "quantize": 8}`. Use a saved quantized model from `GET /v1/models/saved` when one exists.
2. Generate: `POST /v1/generate` with `{"prompt": "...", "seed": [1]}`. Give more seeds for more images.
3. Both calls return a job. Follow it with `GET /v1/jobs/<id>`, or with the event stream `GET /v1/jobs/<id>/events`.
4. When the job ends with `succeeded`, each image is in `result.images[]`, with `path` and `url`.
5. With MCP, call `generate_<model>`, for example `generate_z_image_turbo`. It loads the model if necessary. If it returns a job that has not ended, call `get_job` with `wait_seconds`.

## Image to image

1. Upload the image: `POST /v1/uploads` (multipart field `file`). It returns `upload_id`.
2. Send `image_path: "<upload_id>"` and `image_strength` in the generate request. A higher `image_strength` keeps more of the input image.

## Manage models

| Task | REST | MCP |
|---|---|---|
| Models in the Hugging Face cache | `GET /v1/models/huggingface` | `list_huggingface_models` |
| Download a model | `POST /v1/models/huggingface/download` | `download_model` |
| Save a quantized copy | `POST /v1/models/saved` | `save_model` |
| Saved models | `GET /v1/models/saved` | `list_saved_models` |
| Delete | `DELETE ...?confirm=true` | `delete_*_model` with `confirm: true` |

Ask the user before each delete. A delete frees many gigabytes and cannot be undone.

## Errors

Each error has a `code`. [docs/api.md](../../../docs/api.md) section 1.3 lists them. Report the `code` and the `message` to the user. Do not try the same request again without a change.
