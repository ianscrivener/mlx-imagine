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

1. List the models: `GET /v1/models/mflux/registry` (MCP: `list_registry_models`). Use only a model with `"supported": true`. For other models, tell the user the `reason`.
2. Get the options of the model: `GET /v1/models/mflux/schemas/<name>`.
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

## ControlNet and inpainting

Only a model with `capabilities.modes` that has `controlnet` or `inpaint` can do this. Check `list_registry_models` (`modes`). The models are `qwen-image-2.1-controlnet-union` (and its Turbo version) and `z-image-turbo-controlnet-union-2.1`.

1. Make the control image yourself. The server does not make canny edges, a depth map or a pose image. The image must be ready to use.
2. Give each image to the server: `POST /v1/uploads` (MCP: `upload_image` with base64 text). Use the returned `upload_id`.
3. Qwen-Image 2.1: send `controlnet_image_path` (the control image), or send `image_path` and `mask_image` to inpaint. White in the mask is drawn again. Black stays.
4. Z-Image: send `control`, a list of `"type:upload_id:strength"`, for example `["canny:upl_1a2b3c:0.8"]`. The type is `canny`, `depth`, `hed`, `mlsd` or `pose`. The strength is 0.65 to 1.0.
5. Before a long inpaint, check the mask: `POST /v1/masks/preview` (MCP: `preview_mask`). It draws the white area in red and needs no model.
6. The first load takes long and uses much memory (the base model and the ControlNet). A generate takes minutes: follow the job.

## LoRA files

1. List the LoRA files: `GET /v1/loras` (MCP: `list_loras`). A name is a path inside the LoRA directory.
2. Apply LoRAs at load: `POST /v1/model/load` with `"loras": [{"name": "style.safetensors", "scale": 0.8}]`. To change the LoRAs, load again. A model with `capabilities.supports_lora` false returns `lora_not_supported`.
3. Do not send the options `lora`, `lora_paths` or `lora_style`. The server blocks them.
4. Download a file from Hugging Face: `POST /v1/loras/downloads` with `repo_id` and `filename` (MCP: `download_lora`). Ask the user first. Delete: `DELETE /v1/loras/<name>?confirm=true` (MCP: `delete_lora`).

## Input and output folders

- `output_subdir` in the generate request puts the images in a folder of the output directory.
- `input_subdir` makes a relative `image_path` read from a folder of the input directory. `POST /v1/uploads?subdir=<folder>` saves an upload there.
- A name is a relative path of at most 3 folder names. A name with `..`, a leading `/` or a hidden folder returns an error.

## One server

Only one server runs for each user. If `mlx-imagine serve` stops with "A server already runs", use that server (the error gives the PID). `mlx-imagine mcp` is a stdio client for it. It starts no server.

## Manage models

| Task | REST | MCP |
|---|---|---|
| List models (filters: `location`, `format`, `store`) | `GET /v1/models/mflux` | `list_models` |
| Download a model | `POST /v1/models/mflux/downloads` | `download_model` |
| Save a quantized copy | `POST /v1/models/mflux/saves` | `save_model` |
| Saved models | `GET /v1/models/mflux?store=mflux_dir` | `list_models` |
| Delete | `DELETE ...?confirm=true` | `delete_*_model` with `confirm: true` |

Ask the user before each delete. A delete frees many gigabytes and cannot be undone.

## Errors

Each error has a `code`. [docs/api.md](../../../docs/api.md) section 1.3 lists them. Report the `code` and the `message` to the user. Do not try the same request again without a change.
