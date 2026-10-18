# ROADMAP

Task tracking only. Scope and design are in `docs/`. Sources: [docs/mfluxible-review.md](docs/mfluxible-review.md), [docs/scope.md](docs/scope.md), [docs/api.md](docs/api.md), [MODELS.md](MODELS.md).

| Task | Priority | Feature | Slug |
| :-: | :-: | --- | --- |
| T-26 | 2 | Real-weight LoRA check (`lora` step in the real suite) | `real-lora` |
| T-27 | 2 | Run the service end to end (LaunchAgent, `/mcp`, stdio client, lock) | `service-e2e` |
| T-28 | 4 | Fix docs drift (scope.md, `imagine-api` skill) | `docs-drift` |
| T-29 | 4 | Tests for the LoRA MCP tools | `mcp-lora-tests` |
| T-30 | 5 | Verify phase 0 to 4: PyPI build, release dry run, OpenAPI spec | `phase-check` |
| T-31 | 7 | `service status` shows the PID of the lock owner | `status-pid` |
| T-32 | 8 | CI fails when a stray file is tracked | `stray-files` |


---
## Defer
| Task | Priority | Feature | Slug |
| :-: | :-: | --- | --- |
| T-21 | 8 | Inpainting and `preview_mask` | `inpainting` |



---
## Done
| Task | Priority | Feature | Slug |
| :-: | :-: | --- | --- |
| T-25 | 7 | Per-request input and output subfolders (inside the configured folders) | `io-subdirs` |
| T-12 | 6 | LoRA: list, download, apply | `lora` |
| T-22 | 8 | Startup warning and proxy example | `proxy-example` |
| T-15 | 6 | OpenAI-compatible adapter | `openai-api` |
| T-10 | 5 | Measure MLX cache limit | `mlx-cache-limit` |
| T-17 | 6 | Daily CI against mflux `main` | `ci-mflux-main` |
| T-18 | 6 | Wire up `mflux_models.allow_download` | `mflux-allow-download` |
| T-08 | 3 | Refuse options that a model ignores | `ignored-options` |
| T-11 | 5 | Fix upload sizes for EXIF orientation | `exif-size` |
| T-16 | 6 | Capabilities in `GET /v1/model` | `model-capabilities` |
| T-04 | 1 | Exception text goes into responses | `error-leak` |
| T-06 | 2 | Pure ASGI auth (event stream) | `asgi-auth` |
| T-07 | 2 | Request size limits | `request-limits` |
| T-03 | 1 | MCP generate tool waits too long | `mcp-timeout` |
| T-23 | 9 | MCP setup guide | `mcp-guide` |
| T-01 | 1 | Adapter contract for models without upstream steps | `model-adapters` |
| T-02 | 1 | Flux.3 Klein | `adapter-flux3-klein-turbo` |
| T-05 | 2 | Adapter: Flux2 Klein 9B and Flux2 Klein 9B KV | `adapter-flux2-klein-9b` |
| T-20 | 7 | Adapters: remaining ❌ models in MODELS.md | `adapter-remaining` |
---
## Details

### `model-adapters`
- Models without upstream `validate`/`load`/`generate`: see VLG column in [MODELS.md](MODELS.md).
- Adapter has the same three steps as an mflux command. One file per model. Delete it when mflux ships the steps.
- Needs a change to the "command steps only" rule in AGENTS.md. Ask the maintainer first.

### `ci-mflux-main`
- The daily job `mflux-main` is in `.github/workflows/tests.yml`. The load/unload leak test is the real-weight step `cycles` (by hand only).

### `adapter-remaining`
- Qwen-Image 2.1 and Turbo, Fibo (3), Ming-Image.

### `real-lora`
- Add a `lora` step to `scripts/real/suite.py`. Load `z-image-turbo-q8` with one real LoRA, make an image with a fixed seed, and compare it to the image without the LoRA. They must differ.
- Needs approval from the maintainer: it loads real weights (AGENTS.md rule 1).

### `service-e2e`
- Run `just mac-service install`. Connect Claude Code to `http://127.0.0.1:8765/mcp` and a stdio client to `mlx-imagine mcp`.
- Check: a second `serve` stops with the PID of the first. After a kill of the server, launchd restarts it and the lock is free.

### `docs-drift`
- `docs/scope.md`: LoRA is no longer stage 2 (3.6, 3.16). Section 3.14 says the MCP server runs in the same process and uses stdio with its own engine. Update both.
- `.agents/skills/imagine-api/SKILL.md`: add `loras`, `input_subdir`, `output_subdir` and the single-server lock.

### `mcp-lora-tests`
- `list_loras`, `download_lora` and `delete_lora` have no test in `tests/test_mcp.py`. The REST routes have tests in `tests/test_loras.py`.

### `phase-check`
- Phase 0 to 4 of `docs/scope.md` section 4: run `just pypi-build`, a dry run of `.github/workflows/release.yml`, and check that `/openapi.json` is complete. Then remove "Unchecked" below.

### `status-pid`
- `mlx-imagine service status` shows the PID and the address from `server.lock`, also when the server was not started by the LaunchAgent.

### `stray-files`
- A check in CI or in `just dev` that fails when a file such as `z-tmp.excalidraw` is tracked. The push 9351c08 added one by mistake.

### Unchecked
- Phase 0 to 4 in [docs/scope.md](docs/scope.md) section 4 (CI and PyPI publishing, OpenAPI spec, README). Status not verified here.
