# `mlx-imagine`: Proposal and Scope of Work

Status: stage 1 is built and tested with real weights. Decisions 3.1, 3.2, 3.6, 3.7, 3.8, 3.13 and 3.16 are final.

History: this design started as an mflux plugin (option C in 3.2). It is now a downstream project, because the mflux plugin rules ask for a thin layer and this is a full application. The rule to use only the mflux command steps stays.

`mlx-imagine` is a downstream project of mflux. It uses mflux as a library and makes no change to mflux. It drives models only through these mflux parts: `build_parser()`, the command steps (`validate()`, `load()`, `generate()`), the model registry, `ModelSaver` and the callback system.

## 1. Proposed features

`mlx-imagine` is a new command. It starts a server process that runs until you stop it.

### 1.1 Process model

- The server keeps a maximum of one mflux model in memory.
- The server runs one job at a time. A job is one operation, for example one image generation or one download.
- Clients can load a model, unload a model and generate images.

### 1.2 Model operations

| Operation | Description |
|---|---|
| Load | Load an mflux model into memory. |
| Unload | Free the memory that the loaded model uses. |
| Generate | Make images from text (text-to-image) or from an input image (image-to-image). |
| List mflux models | List the models that `mflux-save` wrote to the mflux model directory. |
| List Hugging Face models | List the compatible models in the local Hugging Face cache. |
| Download | Download a compatible model from Hugging Face. |
| Save and quantize | Write a quantized copy of a model to the mflux model directory. |
| Delete mflux model | Delete a saved model from the mflux model directory. |
| Delete Hugging Face model | Delete a model from the local Hugging Face cache. |

### 1.3 Model coverage

- The goal is to support all standard mflux models. Section 2.1 explains why version 1 cannot reach this goal while it uses only the command steps.
- The server does not support training. `mflux-train` stays a separate command.

### 1.4 Configuration

Clients can read and set these settings:

- The Hugging Face cache directory.
- The mflux model directory.
- Memory limits, for example the MLX memory limit and the MLX cache limit.
- Other general settings, for example the output directory and the default quantization.

### 1.5 Interfaces

- A REST API over HTTP. This is the primary interface.
- An MCP server. AI agents use it to call the same operations.

### 1.6 Packaging

- Package name: `mlx-imagine`. Module: `mlx_imagine`. Command: `mlx-imagine`.
- **Decided:** a name without `mflux` in it, because the package is not part of mflux. The mflux plugin guide also asks packages outside mflux-plugins not to use a bare `mflux-<feature>` name.
- Install with mflux: `uv tool install mflux --with-executables-from mlx-imagine`.
- mlx-imagine is a separate top-level package. It ships no folder named `mflux`.

### 1.7 Service install (macOS only)

- `mlx-imagine service install` installs the server as a macOS LaunchAgent. The server then starts at login and runs in the background.
- `mlx-imagine service uninstall` stops the server and deletes the LaunchAgent.
- `mlx-imagine service status` shows if the LaunchAgent exists and if the server runs.
- On Linux, these commands stop with a clear error.

## 2. Codebase facts that affect the design

These facts come from the mflux code at version 0.23.0.

### 2.1 Model coverage

The repository has 35 image commands. mlx-imagine can drive a command only if the command has a command class with `validate()`, `load()` and `generate()`. mlx-imagine calls only these steps and does not write its own (the same rule as mflux's plugin rule 3).

These 10 commands have the steps. Stage 1 supports all of them.

| Command | Class |
|---|---|
| `mflux-generate-boogu` | `BooguImageCommand` |
| `mflux-generate-ernie-image` | `ErnieImageCommand` |
| `mflux-generate-ernie-image-turbo` | `ErnieImageTurboCommand` |
| `mflux-generate-ideogram4` | `Ideogram4Command` |
| `mflux-generate-krea2` | `Krea2Command` |
| `mflux-generate-lens` | `LensCommand` |
| `mflux-generate-qwen` | `QwenImageCommand` |
| `mflux-generate-qwen-edit` | `QwenImageEditCommand` |
| `mflux-generate-z-image` | `ZImageCommand` |
| `mflux-generate-z-image-turbo` | `ZImageTurboCommand` |

The other commands keep their logic in `main()`. mlx-imagine cannot support them until mflux core adds command steps for them:

- FLUX.1: `mflux-generate` and its in-context, fill, depth, redux and Kontext commands.
- FLUX.2 and FLUX 3: generate and edit.
- Qwen 2.1: generate and edit.
- FIBO: generate and edit, plus `mflux-refine-fibo` and `mflux-inspire-fibo`.
- Ming.
- `mflux-concept` and `mflux-concept-from-image`.
- `mflux-upscale-seedvr2`.

### 2.2 Other facts

1. **The command steps have no stability promise.** [docs/plugins.md](https://github.com/mflux-community/mflux/blob/main/docs/plugins.md) states that the steps "can change in any release".
2. **The memory saver frees model parts.** `CallbackManager.register_callbacks` adds a memory saver. It frees the text encoder, and with `--low-ram` it also frees the transformer. After that, the server cannot use the model again. The Z-Image README tells long-running processes not to use it.
3. **Prompt caches grow.** Qwen, Krea 2 and Ideogram 4 keep `model.prompt_cache`. Each new prompt adds one entry. A long-running process must clear these caches.
4. **mflux reads directory paths at import time.** `MFLUX_CACHE_DIR` in [defaults.py](https://github.com/mflux-community/mflux/blob/main/src/mflux/cli/defaults/defaults.py) and `HF_HUB_CACHE` in `huggingface_hub` are module constants. A setting change at runtime has no effect on code that already imported them.
5. **Saved models do not record their model family.** `ModelSaver` writes `quantization_level` and `mflux_version` only. It does not write the model name or the base model.
6. **mflux has no memory limit setting.** The code calls `mx.set_cache_limit()` for `--low-ram`. No code calls `mx.set_memory_limit()`. mlx-imagine can call both functions directly.
7. **The CLI already describes its own options.** `mflux-capabilities` reads each command's `build_parser()`. It can give the request schema for each command.
8. **The model registry exists.** `AVAILABLE_MODELS` and `MODEL_CLASSES` in [save.py](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/common/cli/save.py) map each model name to its class and its Hugging Face repository.
10. **Callbacks are open to callers.** `model.callbacks.register()` adds a callback to a loaded model. A callback can raise `StopImageGenerationException` to stop a run, as `BatterySaver` does. The callback system has no unregister call.

## 3. Issues and technical decisions

Each item gives a recommendation. Items marked **Decision needed** need your answer before Section 4 is final.

### 3.1 Model coverage gap — **Decided: A**

Version 1 cannot support the 22 commands in Section 2.1, because mlx-imagine makes no change to core. FLUX.1, FLUX.2 and FLUX 3 are in that list.

| Option | Description |
|---|---|
| A. Ship with 13 commands | Version 1 supports only the commands that have steps. Add each family when core adds its steps. |
| B. Copy logic from `main()` | mlx-imagine has its own adapter for each of the 22 commands. This breaks the rule in 2.1. Each mflux release can break the adapters. |
| C. Wait for core | Do not ship until core has steps for all commands. |

Decision: A. mlx-imagine ships with the commands that have steps. The command steps for the other 22 commands are separate core work, outside this scope. Section 4 lists them as a dependency only.

### 3.2 Where mlx-imagine lives — **Decided: C**

The mflux-plugins [CONTRIBUTING.md](https://github.com/mflux-community/mflux-plugins/blob/main/CONTRIBUTING.md) gives 8 criteria for a new plugin. Two of them affect mlx-imagine:

- Criterion 1: mlx-imagine must be a thin layer. The algorithm must be in a separate library on PyPI, with its own tests and releases. A server with a job queue, a REST API and an MCP server is not a thin layer.
- Criterion 2: mlx-imagine must use only commands that have command steps. mlx-imagine agrees with criterion 2.

Other rules in mflux-plugins: a maintainer must agree to the proposal before the first pull request. Each plugin starts from `template/`, has a named maintainer in `CODEOWNERS` and releases through Trusted Publishing.

| Option | Advantages | Disadvantages |
|---|---|---|
| A. All code in mflux-plugins | Users find it in the Plugins table. It uses the shared template, CI and release flow. | It does not agree with criterion 1. The maintainers can refuse it. Every change needs a review from a second code owner. |
| B. A library plus a thin plugin | A library (for example `mlx-imagine-core`, in your own repository) holds the server. A thin plugin in mflux-plugins adds the `mlx-imagine` command. This agrees with criterion 1. | Two packages, two release flows. Most of the code depends on mflux internals, so the library is not truly separate from mflux. |
| C. Your own repository only | You control the code, the reviews and the releases. You can still use the `mflux.extras.server` module name. | You set up CI and PyPI yourself. The Plugins table lists only the folders in mflux-plugins, so users find it less easily. |

Decision: C. mlx-imagine lives in its own repository.

**Update:** mlx-imagine is no longer a plugin. It is a downstream project with its own package, `mlx_imagine`. The text below records the earlier plan.

The goal is still an official mflux plugin. To keep that move easy:

- Use the mflux-plugins `template/` layout and its test rules from the start. Every test docstring starts with `Bug:`. Warnings are errors. No test downloads weights.
- Keep a `CHANGELOG.md`, and state the supported mflux versions in the README.
- When stage 1 is stable, open a "Propose a plugin" issue in mflux-plugins. Ask if the maintainers accept a server as option A or option B.
- The module name does not change after a move, so users do not see the move.

### 3.3 Version compatibility

- Give mflux a lower bound only: `mflux>=0.23`. mlx-imagine guide forbids an upper bound. With a cap, `uv tool install mflux --with-executables-from` picks an older mflux and does not tell the user.
- Run mlx-imagine tests against the mflux `main` branch every day, as the mflux plugin guide recommends. This finds a change to the command steps before a release.
- `tests/test_mflux_drift.py` records a digest of each private mflux function that the server calls or copies. A change in mflux makes the test fail.
- Release a new mlx-imagine version when a new mflux release breaks it.

### 3.4 Web framework and dependencies

- Recommendation: FastAPI with Uvicorn for REST. Use the official `mcp` Python SDK for MCP.
- These dependencies belong to mlx-imagine only. mflux core gets no new dependencies.
- Alternative: the Python standard library `http.server`. It adds no dependencies, but you must write request checks, streaming and an OpenAPI schema yourself.

### 3.5 Model adapter registry

- mlx-imagine keeps a registry that maps each model name to its command class.
- Build the registry from `AVAILABLE_MODELS` and from the command classes in Section 2.1.
- Each registry entry also holds the unload steps for that model family (3.8).
- A model without a command class shows as "not supported" in the model list. It does not cause an error at start.

### 3.6 Request format — **Decided: C**

| Option | Description |
|---|---|
| A. CLI arguments | The request body uses the same option names as the CLI (`steps`, `guidance`, `image_path`). The server builds an `argparse.Namespace` from `build_parser()`. |
| B. Typed schema per model | mlx-imagine writes one Pydantic model for each model family. |
| C. CLI arguments with a generated schema | As A. mlx-imagine also makes a JSON Schema for each model from `build_parser()`, as `mflux-capabilities` does. |

Comparison:

| Topic | A | B | C |
|---|---|---|---|
| A new mflux option works in the server | Yes, with no change to mlx-imagine | No, mlx-imagine must add it | Yes, with no change to mlx-imagine |
| Schema for clients (OpenAPI, MCP `inputSchema`) | No | Yes | Yes |
| Error messages | Refer to CLI flags | Refer to JSON fields | Refer to JSON fields, after the schema check |
| Code that mlx-imagine keeps | Small | Large, and it drifts from mflux | Medium |

These problems apply to A and C:

1. `argparse` calls `sys.exit()` when an option is not correct. The server must catch `SystemExit` and return HTTP 422.
2. Some CLI options are not safe or have no meaning in a server. Examples are `--output`, `--prompt-file`, `--config-from-metadata`, `--stepwise-image-output-dir`, `--battery-percentage-stop-limit` and `--model-path` outside the model directory. mlx-imagine needs a list of blocked options.
3. Some options change which model loads, for example `--model`, `--quantize`, `--base-model` and `--low-ram`. These belong to the load request, not to the generate request.
4. The parser resolves LoRA names when it parses, and this can download a LoRA. LoRA is in stage 2, so stage 1 blocks the LoRA options.
5. `--seed` takes a list. **Decided:** one request can make many images, one for each seed. The server runs them as one job and loads the model once. The job result lists all images. A cancel stops the images that are not complete. The images that are complete stay on disk.

Decision: C. One generic `generate` operation takes the CLI option names. `GET /models/{name}/schema` returns the generated schema for the loaded model. The MCP server has one `generate` tool for each supported model, each with its generated `inputSchema`. This gives agents a clear schema without code that drifts from mflux.

### 3.7 Jobs, progress and cancel

Generation and download can take many minutes. HTTP clients and MCP clients can time out.

- Recommendation: use an asynchronous job model.
  1. The client sends a request. The server returns a job ID at once.
  2. The client reads the job status, or it receives progress events through Server-Sent Events.
  3. The client gets the result when the job is complete.
- One worker thread runs all MLX work. The HTTP thread does no MLX work.
- **Decided:** a queue holds the jobs that wait. The server runs them in order. A client can cancel a job that waits.
- Cancel: a server callback raises `StopImageGenerationException` at the next step. A download cancel needs a separate method.
- Progress: a server callback reports the step number. Register it once for each loaded model, because the callback system has no unregister call.

### 3.8 Memory control

- Do not call `CallbackManager.register_callbacks`, because its memory saver frees model parts (fact 2).
- Unload must delete the model, clear the prompt caches, then call `gc.collect()` and `mx.clear_cache()`.
- Add tests that load and unload a model many times and record the memory after each cycle. A leak is the most probable failure of a long-running server.
- Memory limits use `mx.set_memory_limit()` and `mx.set_cache_limit()`. Test them on 16 GB, 36 GB and 64 GB machines.
- The mflux plugin guide tells plugins not to set MLX memory limits, because the limits belong to the process owner. mlx-imagine is the process owner, so it sets them.
- In low-RAM mode, the mflux memory saver sets a 1 GB cache limit for the process. After each low-RAM job, the server sets its own limits again.
- `CallbackManager.register_callbacks` is never called. Its memory saver frees the text encoder after each run, also outside low-RAM mode. The server calls the two private helpers it needs (`_apply_vae_tiling` and, in low-RAM mode only, `_register_memory_saver`).
- **Decided:** the server supports low-RAM mode. Low-RAM mode frees parts of the model during a run. In low-RAM mode, the server uses the mflux memory saver for each job. After the job, the server marks the model as used up. The next job loads the model again. Each low-RAM job therefore has the full load time.
- The server must check the memory before it loads a model. It must return a clear error, not crash the machine.

### 3.9 Settings at runtime

Because of fact 4, a directory change at runtime is not reliable.

- Recommendation: mlx-imagine sets `MFLUX_CACHE_DIR` and `HF_HOME` from its config file before it imports mflux. A change to a directory setting needs a server restart. The API reports which settings need a restart.
- Memory limits and output settings can change at runtime.
- Config file: `~/.config/mlx-imagine/config.yaml`. A file with the suffix `.toml` is read as TOML. mlx-imagine uses `pyyaml` and `toml`, which mflux already installs.

### 3.10 Model list and model identity

- Hugging Face list: use `huggingface_hub.scan_cache_dir()`. Show only the repositories in `AVAILABLE_MODELS`.
- mflux list: scan the mflux model directory for `model.safetensors.index.json`.
- Because of fact 5, the index file does not name the model family. Recommendation: when mlx-imagine saves a model, it also writes a sidecar file, `mlx-imagine.json`, with `model_name`, `base_model` and the save date.
- A model that `mflux-save` wrote has no sidecar file. For these models, the client must give the base model, as `--base-model` does now. The server then writes the sidecar file.

### 3.11 Delete operations

- Delete Hugging Face model: use `huggingface_hub` `delete_revisions()`. Do not delete directories directly.
- Delete mflux model: delete only directories inside the configured mflux model directory. Resolve the path and refuse any path outside it.
- Refuse to delete the model that is in memory.
- A delete request must include `"confirm": true`.

### 3.12 Security

- Bind to `127.0.0.1` by default. A different address requires a flag and an API token.
- The API can read files, write files and delete files. Limit all paths to the configured directories (input, output, models).
- Image-to-image input: accept a file upload or a base64 string. Accept a local path only inside the input directory.
- Do not log the Hugging Face token. Read it from the normal `huggingface_hub` location.

### 3.13 Image output

- Save each image to the output directory, with mflux metadata, as the CLI does now.
- Return the path and a URL to get the image. The MCP tool also returns the image as image content.
- **Decided:** the server keeps all images on disk. Image management (list, delete, clean-up) is out of scope.

### 3.14 MCP design

- Run the MCP server in the same process. It uses the same job queue and the same loaded model.
- Support stdio (for Claude Desktop and Claude Code) and Streamable HTTP.
- MCP tools: one tool for each operation in 1.2, plus `get_job` and `cancel_job`.
- Delete tools need an explicit `confirm` argument. MCP tool annotations mark them as destructive.

### 3.15 Platforms

- mflux runs on macOS and on Linux with CUDA. Recommendation: support both. Test on macOS first.
- The service install in 1.7 is for macOS only.
- The CI runners cannot load real models. mlx-imagine tests must use tiny models or fake models.

### 3.17 Service install with `launchd` (macOS only)

- `install` writes a property list to `~/Library/LaunchAgents/<label>.plist`. It then runs `launchctl bootstrap gui/<uid>` with that file.
- `uninstall` runs `launchctl bootout gui/<uid>/<label>`, then deletes the property list file.
- `status` runs `launchctl print gui/<uid>/<label>` and calls the server health endpoint.
- Label: a reverse-DNS name, for example `io.github.ianscrivener.mlx-imagine`.
- The property list uses the absolute path of the `mlx-imagine` executable. If the user installs mflux again in a different location, the user must run `install` again.
- The property list sets the config file path and the address. It does not hold the Hugging Face token or the API token.
- Logs go to `~/Library/Logs/mlx-imagine/`.
- `RunAtLoad` is true. `KeepAlive` restarts the server only after a crash, with `ThrottleInterval` set. Without the interval, a server that crashes at start (for example, out of memory) restarts again and again.
- The server starts with no model loaded. A model loads only when a client asks for it. A server that starts at login therefore uses little memory.
- `install` refuses to run when the LaunchAgent is already installed, unless the user gives `--force`.
- Tests: write the property list to a temporary directory and replace the `launchctl` calls with fakes. No test changes the real LaunchAgents of the user.
- Service install is a CLI command only. The REST API and the MCP server cannot install or uninstall the service.

### 3.16 Scope edges — **Decided**

Stage 1 does not include these. They are in stage 2:

- LoRA files (list, download, apply at generation).

Out of scope (see [NON-GOALS.md](NON-GOALS.md)):

- ControlNet commands.
- A step preview on the event stream.
- A cache of the quantized copy at first load.
- An AUTOMATIC1111 API shim, and a separate MCP package.
- Training.
- Image management (3.13).
- A web user interface.

## 4. Draft scope of work

This plan is provisional. It assumes the recommendations in Section 3. All work is in mlx-imagine repository. No phase changes mflux core.

### Phase 0: Set-up

- Create the repository (3.2).
- Set up CI and PyPI publishing.
- Write the API specification as OpenAPI.

Size: small.

### Phase 1: Server core

- Model adapter registry for the 10 stage 1 commands (3.5).
- Job queue, worker thread, job status, cancel (3.7).
- Model manager: load, unload, memory check (3.8).
- Config file and settings API (3.9).
- Service install, uninstall and status for macOS (3.17).
- Model list, download, save and quantize, delete (3.10, 3.11).

Size: medium.

### Phase 2: REST API

- FastAPI routes, request checks from `build_parser()`, Server-Sent Events for progress.
- Token auth and path limits.

Size: medium.

### Phase 3: MCP server

- MCP tools on top of the server core. Support stdio and Streamable HTTP.

Size: small. Most logic is in Phase 1.

### Phase 4: Tests, CI and documentation

- Fast tests with tiny or fake models for the job queue, the API and the path checks.
- A daily CI job against the mflux `main` branch (3.3).
- Slow tests for real load and unload cycles and memory leaks. Run these by hand only.
- A README for the command and an MCP setup guide.

Size: medium.

### Stage 2

- LoRA: list, download and apply at generation. Allow the LoRA options again (3.6).

Size: medium.

### Dependency outside this scope

Support for the 22 commands in Section 2.1 needs command steps in mflux core. That work is not part of this plan. When core adds the steps for a model family, mlx-imagine adds one registry entry and its unload steps. Each addition is small.

## 5. Impact

| Area | Impact |
|---|---|
| mflux core | No change. |
| mflux users | No new dependencies, unless they install mlx-imagine. |
| mlx-imagine repository | A new package, `mlx-imagine`, with its own tests and CI. |
| Maintenance | A new release for each mflux minor release. An update when the command steps change. |

## 6. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Command steps change in an mflux release | The server breaks. | Pin the mflux version. Run daily tests against mflux `main`. |
| Version 1 does not support FLUX | Fewer users find mlx-imagine useful. | State the supported models in the README. Add families as core adds steps. |
| Memory leaks after many load and unload cycles | The machine runs out of memory. | Load and unload tests. A memory check before each load. |
| Real-model tests crash the developer machine | Lost work. | Run real-model tests by hand only, after you close other apps. |
| The API is open on the network | Others can delete files or run jobs. | Bind to localhost by default. Require a token on other addresses. |
| The LaunchAgent restarts a server that crashes at start | The server restarts again and again and uses memory. | Restart only after a crash, with `ThrottleInterval` set. Start with no model loaded. |

## 7. Open questions

None. All decisions are final.
