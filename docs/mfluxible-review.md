# What mlx-imagine can learn from mfluxible

Status: review notes, 2026-10-17. Source: `/Users/ianscrivener/MFlux/mfluxible` at commit `4b94764`, read through its code, its `design/` notes and its GitNexus index.

## Summary

mfluxible and mlx-imagine solve different halves of the same problem.

- **mfluxible** runs one model per process, chosen at start. It puts its effort into what a client sees: streamed steps, inpainting and an OpenAI-compatible API.
- **mlx-imagine** manages models at runtime: load, unload, download, save, delete, a job queue, cancel and a LaunchAgent.

The most valuable things to take from mfluxible are not its features. They are the bugs it already found and measured. Four of them apply to mlx-imagine now (section 2).

## 1. Features that mfluxible has and mlx-imagine does not have

| Feature | What mfluxible does | Value for mlx-imagine | Recommendation |
|---|---|---|---|
| OpenAI-compatible API | `/v1/images/generations`, `/v1/images/edits`, `/v1/models`, with streaming and partial images. | High. Many clients speak this API. | Stage 2. Add a thin adapter over the job queue. |
| Chat stub | A chat-completions endpoint that answers with a `generate_image` tool call. | Low | Skip. |
| Inpainting | `mask` (PNG), `mask_feather`, and `mask_boxes` as fractions of the frame. A masked request defaults `image_strength` to 0.0. Needs a custom scheduler. | Medium | Later. mflux has fill and Kontext commands, but they have no command steps yet. |
| Fractional start | A custom scheduler starts img2img from a fraction of a step. | Low | Skip. It needs a scheduler of our own. |
| Object detection | A VLM finds objects and returns boxes, from a local model or a sidecar. | Low | Skip. |
| Terminal clients | Python and Node scripts that stream and show images inline in iTerm2. | Low | Skip. `curl` and MCP cover this. |
| `preview_mask` MCP tool | Draws the mask over the image with no model, so an agent can check its boxes. | Low | Only with inpainting. |
| LoRA | Bakes LoRAs in at load. | High (already in stage 2) | Copy the bake rules when LoRA arrives. |
| More model families | 15 checkpoints, FLUX.1 and FLUX.2 Klein included. It calls the mflux model classes directly. | High, but it bypasses the command steps | Keep our rule: use only the command steps. mfluxible's `models.py` is a good list of the per-model differences to check when mflux adds command steps. |
| Basic auth and a proxy example | In-app Basic auth, plus `Caddyfile.example` with a bearer token. | Low. We have a bearer token. | Copy the startup warning and the proxy example only. |
| `/health` lists capabilities | `default_steps`, `supports_guidance`, `default_guidance`, `supports_negative_prompt`, `supports_mask`. Clients read it to choose defaults. | Medium | Add these to `GET /v1/model`. Our schema route has part of this. |

## 2. Problems in mlx-imagine that mfluxible already found

These are in priority order. Each one is a change to code that exists now.

### 2.1 The MCP generate tool waits too long (high)

mfluxible measured a Claude Code tool call that failed at about 60 seconds, although the server sent a progress notification every 8 seconds. Progress notifications do not keep a host's timeout open. The limits change with each host and each version.

mlx-imagine's generate tools wait up to `wait_seconds = 300` by default. A long generate will fail in Claude Code, although the job continues on the server.

Fix: set the default `wait_seconds` to 45. After that time, return the job ID and tell the agent to call `get_job`. mfluxible does the same thing with its `check_image` tool.

### 2.2 Exception text goes into responses (high)

mfluxible's rule: no path from an exception to a response body. Text from mflux, MLX and Pillow quotes absolute cache paths. mfluxible logs the exception and sends a fixed sentence. A structural test reads the source with `ast` and fails if a route binds a caught exception to a name.

mlx-imagine puts `str(exc)` and a traceback in `job.error` (`jobs.py`), and every client can read it.

Fix:
- Keep the full text in the server log.
- Send a fixed message and an error code.
- Keep `ApiError` messages, because the server writes those.
- Add an `ast` test for `api.py` and `mcp_server.py`.

### 2.3 The auth middleware can hold back the event stream (medium)

mfluxible uses pure ASGI middleware for auth and for the body limit. Its reason: Starlette's `BaseHTTPMiddleware` passes the response through a memory stream, and that is the wrong shape for SSE.

mlx-imagine's auth uses `@app.middleware("http")`, which is a `BaseHTTPMiddleware`. The SSE test passes with `TestClient`, but no test makes sure that events reach a real client one at a time.

Fix: write the auth check as a pure ASGI middleware. Add a test that reads the first SSE event while the job still runs.

### 2.4 Requests have no size limits (medium)

mfluxible limits the request body (64 MB, pure ASGI, checked before it reads the body) and the numeric fields.

mlx-imagine has no limit on `width`, `height`, `steps` or the number of seeds. A request for 8192×8192, or 1000 seeds, is accepted and can use all the memory.

Fix: add config settings for the maximum pixels, steps and seeds. Check them in `submit_generate`, before the job enters the queue.

### 2.5 Options that a model ignores are accepted (medium)

mfluxible returns 400 for `guidance` or `negative_prompt` when the model cannot use them. It checks the *effective* guidance: most CFG models encode the negative prompt only above guidance 1.0, and Krea-2's default is exactly 1.0.

mlx-imagine leaves ignored options out of the schema, but a request that sends them is accepted. mflux then ignores them with no message.

Fix: read each command's `IGNORED_OPTIONS` and `CONDITIONAL_OPTIONS`, as `options.schemas()` does already, and refuse those options with `option_ignored`. Copy mfluxible's effective-guidance check for the negative prompt.

### 2.6 The MLX cache limit is a guess (low)

mfluxible measured a server that became 15 times slower over three runs. The cause was MLX's buffer cache, which grew until macOS paged the weights out. The default cache limit of 1 GB stopped it.

mlx-imagine's default is 4 GB, from one run.

Fix: measure 1 GB and 4 GB over ten jobs in one process, then choose. Measure in a fresh process each time.

### 2.7 Upload sizes ignore EXIF orientation (low)

mflux turns an input image by its EXIF orientation before it encodes it. mfluxible checks sizes after that turn. mlx-imagine's upload response gives the size before the turn. A phone photo then reports width and height the wrong way round.

Fix: report the oriented size, as `mflux.utils.exif_orientation.oriented_size` gives it.

## 3. Development patterns to borrow

| Pattern | In mfluxible | How to use it here |
|---|---|---|
| Rules in one file, reasons in another | `CLAUDE.md` gives each rule in one line and names a `design/<area>.md` file. That file has the measurements and the dead ends. A test checks that the two point at each other. | Move the reasons out of `docs/scope.md` into `design/` files. Keep `AGENTS.md` as the rule list. |
| Tests for documentation drift | `test_docs.py` checks that the documented `/health` sample lists every model, and that the version in it is current. Config tests fail when `config.KNOWN`, the example config and `docs/server.md` disagree. | Add a test that checks the settings table in `docs/api.md` against `config.FIELDS`. Add `config.example.yaml` and test it the same way. |
| Structural tests that read the source | An `ast` test makes sure that no route binds a caught exception. | Use it for 2.2. Use the same idea for "no MLX call outside the worker thread". |
| A test double built from mflux's real parts | `ToyModel` uses mflux's real `CallbackRegistry`, `Config`, `VAEUtil` and `ImageUtil`. Only the VAE and the prediction are fake. | Our `FakeModel` and `FakeImage` skip `Config` and `GeneratedImage`. Build the fake image with `ImageUtil.to_image`, as the mflux-plugins template does, so the metadata and save code are real. |
| Validation returns a reason | `request_problem()` returns a string or `None`. The route turns it into a 400 before the stream starts. | We raise `ApiError`, which is equal in effect. Keep the rule that every check runs before the job enters the queue. |
| Settings in one place | `runtime.py` holds the settings and the engine. Routes read `runtime.NAME` when they run, so a test patches one place. | Our `engine.settings` does this already. |
| Routes split by surface | `routes/native.py`, `openai.py`, `misc.py`. | Split `api.py` when the OpenAI adapter arrives. |
| Startup warning | The server logs a warning when it binds to a network address with no auth. | We refuse to start in that case, which is stricter. Keep it. |
| CI that matches the hardware | Tests run on `macos-14`, because MLX has no Linux wheel for this use. Lint runs on `ubuntu-latest`. A build job checks the wheel. `permissions: contents: read` is spelled out. uv's cache keeps the install fast. | Copy this layout for the first CI workflow. |
| Clients that ask before they offer | The MCP tool reads `/health` and treat a missing field as "unknown, let the server decide". They send `steps` and `guidance` as null, so the loaded model chooses. | Tell clients to do the same. Our MCP tools already omit unset values. |

## 4. Things mlx-imagine does better

These are worth keeping. Some of them are worth offering back to mfluxible.

- **Cancel works.** mfluxible's design notes say that mflux cannot stop `generate_image()` from outside. mlx-imagine stops it at the next step with a callback that raises `StopImageGenerationException`. The real-weight test showed this. mfluxible could use the same callback for a client that disconnects.
- **Models change at runtime.** mfluxible needs a restart to change the model.
- **Model management.** Download, save, quantize, delete, limits and allow flags.
- **A queue with job IDs.** mfluxible serves one request at a time and holds the HTTP connection for the whole run.
- **The request schema comes from mflux.** Our options come from each command's `build_parser()`, so a new mflux option works with no change. mfluxible keeps its own Pydantic schema and per-model table.
- **Low-RAM mode and a LaunchAgent.**

## 5. Proposed order

1. Fix 2.1 (MCP wait) and 2.2 (exception text). Both are small, and both affect users now.
2. Fix 2.3 (pure ASGI auth) and 2.4 (request limits).
3. Fix 2.5 (ignored options).
4. Measure 2.6 (cache limit).
5. Then the stage 2 features: LoRA and the OpenAI adapter.
