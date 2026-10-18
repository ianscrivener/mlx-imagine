# AGENTS.md

Instructions for coding agents that work on `mlx-imagine`. This file is the one source of the shared agent rules. [docs/development/agent-setup.md](docs/development/agent-setup.md) tells where each agent file goes.

## Package

- mlx-imagine is a downstream project of mflux. It uses mflux as a library. It is not an mflux plugin.
- All code goes in `src/mlx_imagine/`. Do not add a folder named `mflux`: it would shadow the library. `tests/test_package.py` checks this.
- Inside the package, use relative imports.

## Skills

The skills are in `.agents/skills/`. `.claude/skills/` and `.cursor/skills/` hold symlinks to them.

| Task | Skill |
|---|---|
| Setup, uv, justfile recipes, mflux version switches | `imagine-dev-env` |
| Tests, the drift test, real-weight checks | `imagine-testing` |
| Commit, push and pull request | `imagine-pr` |
| Release and PyPI | `imagine-release` |
| Make images through a running server | `imagine-api` |
| Text that people or agents read | `asd-ste100` |

## Commands

| Task | Command |
|---|---|
| Install | `just venv-install` |
| Lint, format check, type check, docs lint | `just dev` |
| All checks | `just dev-check` |
| Tests | `just test` |
| Real-weight checks | `just test-real` (ask the maintainer first) |

Run `just dev-check` before you call a change done.

## Rules for the code

1. Do not load real weights in a test. Do not start a real generation without the maintainer's approval: a real model can use all the memory of the machine.
2. Call mflux's `build_parser()` and command steps. Do not write your own version of them. Do not copy a command's `main()`. Exception: a model whose mflux command has no command steps gets an adapter in `src/mlx_imagine/adapters/`. An adapter has the same `build_parser()`, `validate`, `load` and `generate` as an mflux command. Delete it when mflux ships the steps.
3. If you must copy mflux code (an adapter does), mark the copy with a comment, list it in `NOTICE`, and pin it in `tests/test_mflux_drift.py`. Add a row to [docs/mflux-internals.md](docs/mflux-internals.md) for each new part of mflux that the code uses.
4. Do all MLX work on the job queue's worker thread. The HTTP and MCP threads only add, read and cancel jobs.
5. Never call `CallbackManager.register_callbacks`. Its memory saver frees model parts after each run.
6. Do not read the prompt with `PromptUtil.read_prompt`. The prompt `-` reads stdin, which is the MCP stdio channel.
7. `config.py` must not import mflux. mflux and huggingface_hub read their directory settings once, at import.
8. Each test docstring starts with `Bug:` and names the bug that makes the test fail. Warnings are errors.
9. The REST route and the MCP tool of one operation call the same engine method. Do not put logic in only one of them.
10. Check each request before its job goes into the queue. A request that waits in the queue must not fail for a reason that the server could see at once.

## Code style

- Use type hints on each public function.
- Add a comment only when the reason is not clear from the code. Keep it short.
- Put a fact in one place. The defaults of a model come from mflux, and the settings come from `config.FIELDS`.
- When the API, an option or a setting changes, change `docs/api.md` and `README.md` in the same commit.
- Write text that people or agents read in ASD-STE100. Use the `asd-ste100` skill: strict for error messages, log messages and agent instructions, STE-flavored for docs, comments and PR text. Run `just dev-docs` for the Markdown files.

## Git

- Do not push or force-push without the maintainer's approval for each push.
- Before a push, ask: "Do we run CodeRabbit locally first?"
- A local commit is always permitted.
- Before you squash or rewrite the branch history, make a local backup branch: `backup/<branch>-pre-squash-<YYYYMMDD-HHMM>`. Do not push it.

## CI

- Pin each `uses:` to a full commit SHA, with the release version in a comment: `uses: actions/checkout@<sha>  # v7.0.1`. Do not use a tag such as `@v7`: a tag can move.
- To update an action, pin the latest release of the same major version. Change a major version in its own PR.
- The ruff version is pinned in `pyproject.toml`. The CI lint job uses the same version. Change both together.

## Work method

- For work on many files or with high risk, write a short plan first. Give the goal, the limits, the files, and how you will check the result. Ask for approval before you write code.
- If the scope of the task changes, stop and ask.
- Prefer a result that a command can check (tests, lint, build) to a guess.

## Report a bug

Use one scenario for each problem. Write the steps in order and end with one line `Fix:`.

```text
Scenario: Save name crashes the server
1) A client sends POST /v8/models/saved with "name": 123.
2) safe_name runs `"/" in name` and raises TypeError.
3) The client gets HTTP 500.
Fix: Refuse a name that is not a string with HTTP 400.
```

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
| `adapters/` | Commands for models that have no mflux command steps yet. |
| `options.py` | Request fields to command-line options, and the JSON Schemas. |
| `jobs.py` | The job queue and the worker thread. |
| `engine.py` | The loaded model and the job handlers. |
| `hf_models.py`, `saved_models.py` | The Hugging Face cache and the saved models. |
| `api.py`, `mcp_server.py` | The REST API and the MCP tools. |
| `service.py` | The macOS LaunchAgent. |
