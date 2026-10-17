# ROADMAP

Task tracking only. Scope and design are in `docs/`. Sources: [docs/mfluxible-review.md](docs/mfluxible-review.md), [docs/scope.md](docs/scope.md), [docs/api.md](docs/api.md), [MODELS.md](MODELS.md).

| Task | Priority | Feature | Slug |
| :-: | :-: | --- | --- |
| T-01 | 1 | Adapter contract for models without upstream steps ✅ | `model-adapters` |
| T-02 | 1 | Flux.3 Klein Turbo ✅ (not run with real weights) | `adapter-flux3-klein-turbo` |
| T-03 | 1 | MCP generate tool waits too long | `mcp-timeout` |
| T-04 | 1 | Exception text goes into responses | `error-leak` |
| T-05 | 2 | Adapter: Flux2 Klein 9B and Flux2 Klein 9B KV | `adapter-flux2-klein-9b` |
| T-06 | 2 | Pure ASGI auth (event stream) | `asgi-auth` |
| T-07 | 2 | Request size limits | `request-limits` |
| T-08 | 3 | Refuse options that a model ignores | `ignored-options` |
| T-10 | 5 | Measure MLX cache limit | `mlx-cache-limit` |
| T-11 | 5 | Fix upload sizes for EXIF orientation | `exif-size` |
| T-12 | 6 | LoRA: list, download, apply | `lora` |
| T-15 | 6 | OpenAI-compatible adapter | `openai-api` |
| T-16 | 6 | Capabilities in `GET /v1/model` | `model-capabilities` |
| T-17 | 6 | Daily CI against mflux `main` | `ci-mflux-main` |
| T-18 | 6 | Wire up `mflux_models.allow_download` | `mflux-allow-download` |
| T-20 | 7 | Adapters: remaining ❌ models in MODELS.md | `adapter-remaining` |
| T-21 | 8 | Inpainting and `preview_mask` | `inpainting` |
| T-22 | 8 | Startup warning and proxy example | `proxy-example` |
| T-23 | 9 | MCP setup guide | `mcp-guide` |

## Details

### `model-adapters`
- Models without upstream `validate`/`load`/`generate`: see VLG column in [MODELS.md](MODELS.md).
- Adapter has the same three steps as an mflux command. One file per model. Delete it when mflux ships the steps.
- Needs a change to the "command steps only" rule in AGENTS.md. Ask the maintainer first.

### `ci-mflux-main`
- Also: slow tests for load/unload cycles and memory leaks. Run by hand only.

### `adapter-remaining`
- Qwen-Image 2.1 and Turbo, Fibo (3), Ming-Image, Flux.2 Klein Base 9B.

### Unchecked
- Phase 0 to 4 in [docs/scope.md](docs/scope.md) section 4 (CI and PyPI publishing, OpenAPI spec, README). Status not verified here.
