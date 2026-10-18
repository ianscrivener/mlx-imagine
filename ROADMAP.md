# ROADMAP

Task tracking only. Scope and design are in `docs/`. Sources: [docs/mfluxible-review.md](docs/mfluxible-review.md), [docs/scope.md](docs/scope.md), [docs/api.md](docs/api.md), [MODELS.md](MODELS.md).

| Task | Priority | Feature | Slug |
| :-: | :-: | --- | --- |

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
| T-27 | 2 | Run the service end to end (LaunchAgent, `/mcp`, stdio client, lock) | `service-e2e` |
| T-26 | 2 | Real-weight LoRA check (`lora` step in the real suite) | `real-lora` |
| T-32 | 8 | CI fails when a stray file is tracked | `stray-files` |
| T-31 | 7 | `service status` shows the PID of the lock owner | `status-pid` |
| T-30 | 5 | Verify phase 0 to 4: PyPI build, release dry run, OpenAPI spec | `phase-check` |
| T-29 | 4 | Tests for the LoRA MCP tools | `mcp-lora-tests` |
| T-28 | 4 | Fix docs drift (scope.md, `imagine-api` skill) | `docs-drift` |
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
- Done: Qwen-Image 2.1 and Turbo (`adapters/qwen21.py`). Left: Fibo (3), Ming-Image.

### `real-lora`
- Run on 2026-10-18 with the saved model `Qwen--Qwen-Image-2.1-mflux-q8` (adapter `qwen21`) and 3 LoRAs for Qwen-Image 2.1: all 3 change the image with the same seed. At scale 1.0 and 12 steps two of them gave distorted faces. At scale 0.6 and 28 steps (`LORA_SCALE`, `LORA_STEPS`) all gave clean images.
- Run: `LORAS=a.safetensors,b.safetensors LORA_BASE_MODEL=qwen-image-2.1 zsh scripts/real/run.sh lora <model> <image> config.yaml`. Stop the service first: a server that runs holds the lock.

### `service-e2e`
- Run on 2026-10-18 with the LaunchAgent: `/mcp` answers, the stdio client lists 37 tools, a second `serve` stops with the PID of the first, and after `kill -9` launchd starts a new server (new PID in the lock file).

### `phase-check`
- Checked on 2026-10-18: `just pypi-build` makes the wheel and the sdist (the wheel has the `mlx-imagine` command and all modules). `/openapi.json` lists 25 paths, each with a summary, but no operation has a description.
- The release workflow stops at its CHANGELOG gate: `CHANGELOG.md` has no `## [0.0.1]` section yet. This is expected before a release.
- Not checked: the PyPI Trusted Publisher and the `pypi` environment on GitHub. Only the maintainer can check them.
