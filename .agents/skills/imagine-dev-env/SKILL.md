---
name: imagine-dev-env
description: Set up and work in the mlx-imagine dev environment (macOS arm64, uv, justfile recipes, lint, format, type check, mflux version switches). Use for setup problems and to find the correct command for a task.
---

# mlx-imagine dev environment

mlx-imagine runs on macOS arm64. It uses `uv` for Python and `just` for each workflow.

## When to use

- You set up the repository, or an environment problem occurs.
- You need the correct command to lint, format, type check, test, build or serve.
- You need to run mlx-imagine against a different mflux version.

## Requirements

- `just` 1.50 or later (`brew install just`). CI checks the justfile format with `just justfile-check`.
- `uv`. Do not use `pip` in this repository.

## Instructions

1. Run `just` to see each recipe.
2. Make the environment with `just venv-install`. It installs the locked dependencies into `.venv`.
3. Use a recipe when one exists:

   | Task | Recipe |
   |---|---|
   | Lint, format check, type check, docs lint | `just dev` |
   | Format the code and apply the safe lint changes | `just dev-fix` |
   | All checks before a commit | `just dev-check` |
   | Tests with no weights | `just test` |
   | Build the wheel and the sdist | `just pypi-build` |
   | Start the server | `just dev-serve` |
   | Show a running server | `just serve-status` |

4. Run Python with `uv run <command>`. Do not activate a virtualenv by hand.
5. Do not edit `uv.lock` by hand. After a change to `pyproject.toml`, run `uv lock`.

## The justfile and uv

- The justfile sets `UV_NO_SYNC=1`. Because of this, `uv run` does not install the locked mflux again after a switch.
- The justfile removes `VIRTUAL_ENV`. The virtualenv of a different project can be active in the shell. uv must use this project's `.venv`.
- Outside the justfile, run uv as `env -u VIRTUAL_ENV uv ...` for the same reason.

## mflux versions

| Task | Recipe |
|---|---|
| Show the installed mflux and where it came from | `just mflux-version` |
| Install mflux from git `main` | `just mflux-bump` |
| Install mflux from a pull request | `just mflux-pr <number>` |
| Install mflux from a branch | `just mflux-branch <name>` |
| Go back to the locked mflux from PyPI | `just mflux-reset` |

After a switch, run `just test-drift`. If the drift test fails, use the `imagine-testing` skill.

## Type checking

- The type checker is `ty`. Its version is pinned in the `dev` group of `pyproject.toml`.
- Run it with `just dev-types`.
- Do not add an ignore comment to make an error stop. Correct the type. If mflux gives a wrong type, use a narrow `# ty: ignore[<rule>]` and say why in the comment.
