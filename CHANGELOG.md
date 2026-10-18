# Changelog

All notable changes to mlx-imagine are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The Swagger page lists the operations in alphabetical order.
- The MCP tools return a JPEG preview of at most 1024 pixels for each generated image, not the full PNG. The job result has the path of the full image.
- `mlx-imagine mcp` is a thin stdio client of the server that runs. It no longer starts its own engine. The `just serve-mcp` recipe is removed. Run the server as a service (recommended) or with `just dev-serve`.

### Fixed

- A download job (a model or a LoRA) no longer fails with `AttributeError: 'unit'` after the file arrived. The progress hook of the job now counts bytes.
- A load of a model that is not complete in the cache returns `not_found` when the request arrives, not after it waited in the queue. The message names the saved copies of the model.

### Added

- The setting `huggingface_models.max_download_model_size_gb` limits the size of one download, for a model and for a LoRA file. `GET /v1/server` shows it as `permissions.max_download_gb`.
- `GET /v1/server` shows `permissions` and `lora_directory`. `GET /v1/loras` and the MCP tool `list_loras` return the LoRA `directory`. A refused LoRA download says where to put the file. These come from the first customer test (`customer_testing/01_ControlNet`).
- `just mflux-bump`, `mflux-pr`, `mflux-branch` and `mflux-reset` restart the LaunchAgent after the install. The new recipe `mflux-restart` does it, and skips it when the LaunchAgent is not installed.
- The MCP tool `upload_image` (base64 in, upload ID out). A client now learns three things from the MCP alone. It learns how to give images. It learns how to make a control image. It learns what the mask colors mean. The text is in the MCP `instructions`, the description of each ControlNet generate tool, and the schema of each image option.
- ControlNet and inpainting, for the models that support them. The adapter for Qwen-Image 2.1 with the Fun ControlNet Union takes a control image, or an image and a mask to inpaint. The adapter for Z-Image Turbo with the Fun ControlNet Union 2.1 takes `control` specs. The options `controlnet_image_path`, `mask_image` and `control` follow the input directory rules.
- `POST /v1/masks/preview` and the MCP tool `preview_mask` draw a mask on an image, with no model.
- The real-weight steps `inpaint` and `control`. The watchdog reads `MAX_SWAP_GROWTH_MB`.
- An adapter for Qwen-Image 2.1 and Qwen-Image 2.1 Turbo (`adapters/qwen21.py`). It runs the mflux edit command: text to image, and edits with up to 10 reference images (`image_paths`), a mask (`mask_image`) and `strength`.
- `mlx-imagine service status` shows `lock_owner`: the PID and address of the server that holds the lock.
- `just dev-stray` and a CI step that fail when git tracks a scratch file, a weight file or a log.
- The real-weight step `lora` in `scripts/real/suite.py`. It takes `LORAS`, `LORA_BASE_MODEL`, `LORA_QUANTIZE`, `LORA_SCALE` and `LORA_STEPS` from the environment.
- The generate options `input_subdir` and `output_subdir`, and `subdir` for `POST /v1/uploads`. They name a folder inside the input or output directory for one request.
- LoRA: `GET /v1/loras`, `POST /v1/loras/downloads`, `DELETE /v1/loras/{name}`, the MCP tools `list_loras`, `download_lora` and `delete_lora`, and the `loras` field of the load request. The setting `loras.local_directory` holds the files.
- Adapters for mflux models that have no command steps. The adapters run FLUX 3 Image Klein and FLUX.2 Klein.
- A server lock (`~/Library/Application Support/mlx-imagine/server.lock`). A second server stops with an error that gives the PID of the first.
- `mlx-imagine`: a REST API and an MCP server for the 10 mflux commands that have command steps.
- A job queue that runs one job at a time, with progress events and cancel.
- Load, unload and generate, with low-RAM mode.
- Hugging Face cache list, download and delete.
- Saved model list, save with quantization, and delete. The server writes `mlx-imagine.json` next to each saved model.
- A config file and a settings API.
- `mlx-imagine service install | uninstall | status` for a macOS LaunchAgent.
- Agent skills in `.agents/skills/` (`imagine-dev-env`, `imagine-testing`, `imagine-pr`, `imagine-release`, `imagine-api`), with symlinks in `.claude/skills/` and `.cursor/skills/`.
- GitHub Actions: tests and lint, a daily run against mflux `main`, and a CHANGELOG check for each pull request.
- A release workflow that publishes to PyPI through Trusted Publishing.
