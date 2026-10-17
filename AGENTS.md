# AGENTS.md

Instructions for coding agents that work on `mlx-imagine`.

## Package

- mlx-imagine is a downstream project of mflux. It uses mflux as a library. It is not an mflux plugin.
- All code goes in `src/mlx_imagine/`. Do not add a folder named `mflux`: it would shadow the library. `tests/test_package.py` checks this.
- Inside the package, use relative imports.

## Commands

| Task | Command |
|---|---|
| Install | `just venv-install` |
| Tests | `just test` |
| All checks | `just dev-check` |
| Real-weight checks | `just test-real` (ask the maintainer first) |

Run `just dev-check` before you call a change done. The justfile sets `UV_NO_SYNC=1`, so `uv run` keeps an mflux that `just mflux-bump`, `just mflux-pr` or `just mflux-branch` installed. `just mflux-reset` goes back to the locked mflux.

## Rules

1. Do not load real weights in a test. Do not start a real generation without the maintainer's approval: a real model can use all the memory of the machine.
2. Call mflux's `build_parser()` and command steps. Do not write your own version of them. Do not copy a command's `main()`.
3. If you must copy mflux code, mark the copy with a comment, list it in `NOTICE`, and pin it in `tests/test_mflux_drift.py`.
4. Do all MLX work on the job queue's worker thread. The HTTP and MCP threads only add, read and cancel jobs.
5. Never call `CallbackManager.register_callbacks`. Its memory saver frees model parts after each run.
6. Do not read the prompt with `PromptUtil.read_prompt`. The prompt `-` reads stdin, which is the MCP stdio channel.
7. `config.py` must not import mflux. mflux and huggingface_hub read their directory settings once, at import.
8. Each test docstring starts with `Bug:` and names the bug that makes the test fail. Warnings are errors.

## When the drift test fails

1. Read mflux's diff of the function that the failure names.
2. Bring the server in line with it.
3. Record the new digest that the failure prints.

## Code layout

| File | Contents |
|---|---|
| `cli.py` | The `mlx-imagine` command. Sets the environment before mflux loads. |
| `config.py` | Settings and the config file. |
| `registry.py` | Which mflux command runs which model. |
| `options.py` | Request fields to command-line options, and the JSON Schemas. |
| `jobs.py` | The job queue and the worker thread. |
| `engine.py` | The loaded model and the job handlers. |
| `hf_models.py`, `saved_models.py` | The Hugging Face cache and the saved models. |
| `api.py`, `mcp_server.py` | The REST API and the MCP tools. |
| `service.py` | The macOS LaunchAgent. |
