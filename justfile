# justfile for mlx-imagine. Run `just` for the list of recipes.

set shell := ["bash", "-euo", "pipefail", "-c"]

# Another project's virtualenv can be active in the shell. uv must use this project's .venv.
unexport VIRTUAL_ENV

# `uv run` must not sync back to the locked mflux after an mflux-* recipe installed a git version.
# `just venv-install` and `just mflux-reset` sync on purpose.
export UV_NO_SYNC := "1"

python := ".venv/bin/python"
mflux_repo := "https://github.com/mflux-community/mflux"
config := "config.yaml"
real_model := "z-image-turbo-q8"
real_image := "tests/assets/puffin-512.png"
real_steps := "load,seeds,cancel,img2img,mcp,lowram,unload"
ste_lint := ".agents/skills/asd-ste100/scripts/ste-lint.py"

# Show all recipes
default:
    @just --list --unsorted

# Run the dev checks in one go: lint, format check, type check, docs lint, justfile format
dev: dev-lint dev-format-check dev-types dev-docs justfile-check

# Run every check before a commit: the dev checks, then the tests
dev-check: dev test

# Lint the Markdown docs and the skills with the ASD-STE100 linter
dev-docs:
    python3 {{ ste_lint }} README.md AGENTS.md CHANGELOG.md docs/*.md docs/development/*.md .agents/skills/imagine-*/SKILL.md .github/pull_request_template.md

# Format with ruff and apply the safe lint fixes
dev-fix:
    uv run ruff check --fix .
    uv run ruff format .

# Check the format without changes
dev-format-check:
    uv run ruff format --check .

# Lint with ruff (read-only)
dev-lint:
    uv run ruff check .

# Type check with ty
dev-types:
    uv run ty check src

# Check the justfile format (CI runs this)
justfile-check:
    just --fmt --check

# Format the justfile
justfile-fmt:
    just --fmt

# Install mflux from a branch, e.g. just mflux-branch my-feature
mflux-branch name:
    uv pip install --python {{ python }} --reinstall-package mflux "mflux @ git+{{ mflux_repo }}@{{ name }}"
    @just mflux-version

# Install mflux from the latest git main
mflux-bump:
    uv pip install --python {{ python }} --reinstall-package mflux "mflux @ git+{{ mflux_repo }}@main"
    @just mflux-version

# Install mflux from a pull request, e.g. just mflux-pr 886
mflux-pr number:
    uv pip install --python {{ python }} --reinstall-package mflux "mflux @ git+{{ mflux_repo }}@refs/pull/{{ number }}/head"
    @just mflux-version

# Go back to the locked mflux from PyPI
mflux-reset:
    uv sync --locked --reinstall-package mflux
    @just mflux-version

# Show the installed mflux version and where it came from
mflux-version:
    @{{ python }} -c 'import importlib.metadata as m, json; d = m.distribution("mflux"); u = d.read_text("direct_url.json"); s = json.loads(u) if u else {}; v = s.get("vcs_info", {}); print("mflux", d.version, "from", (s.get("url", "") + " @ " + v.get("requested_revision", "") + " (" + v.get("commit_id", "")[:9] + ")") if v else "PyPI (uv.lock)")'

# Build the wheel and the sdist into dist/ for PyPI
pypi-build:
    uv build

# Start the server (REST API and MCP over HTTP) with config.yaml (Ctrl-C to stop)
serve-api config=config:
    uv run mlx-imagine --config {{ config }} serve

# Start the MCP server on stdio with config.yaml
serve-mcp config=config:
    uv run mlx-imagine --config {{ config }} mcp

# Manage the macOS LaunchAgent: start | stop | restart | status | install | uninstall
mac-service action config=config:
    @case "{{ action }}" in start|stop|restart|status|install|uninstall) ;; *) echo "Use: just mac-service start|stop|restart|status|install|uninstall"; exit 1;; esac
    uv run mlx-imagine --config {{ config }} service {{ action }}

# Show the health and state of a running server
serve-status port="8765":
    @curl -sf http://127.0.0.1:{{ port }}/v1/server | {{ python }} -m json.tool || echo "No server at port {{ port }}."

# Run all offline tests (fake models, no weights)
test *args:
    uv run pytest {{ args }}

# Stop at the first failure and show local variables
test-debug *args:
    uv run pytest -x --showlocals {{ args }}

# Run the mflux drift test: fails when an mflux function the server depends on changes
test-drift:
    uv run pytest tests/test_mflux_drift.py

# Run one test file, e.g. just test-file api
test-file name:
    uv run pytest tests/test_{{ name }}.py

# Run the tests whose names match an expression, e.g. just test-k cancel
test-k expression:
    uv run pytest -k "{{ expression }}"

# Real-weight checks under the mem watchdog:load,seeds,cancel,img2img,mcp,lowram,save,unload
[confirm("This loads real model weights and can use most of the memory of this Mac. Close heavy apps first. Continue?")]
test-real steps=real_steps model=real_model:
    zsh scripts/real/run.sh "{{ steps }}" "{{ model }}" "{{ real_image }}" "{{ config }}"

# Show the log of the last real-weight run
test-real-log:
    cat .real-logs/suite.log

# Real-weight save and quantize of ZIT, then delete the save
[confirm("This loads about 20 GB of bf16 weights and writes a 10 GB model. Close heavy apps first. Continue?")]
test-real-save:
    zsh scripts/real/run.sh "save" "{{ real_model }}" "{{ real_image }}" "{{ config }}"

# Delete .venv, caches, build output & real-test logs (keeps input/ and output/)
[confirm("Delete .venv, the caches, dist/ and .real-logs/? Run `just venv-install` after it. Continue?")]
venv-delete:
    rm -rf .venv .pytest_cache .ruff_cache dist .real-logs
    find . -name __pycache__ -type d -not -path "./.venv/*" -prune -exec rm -rf {} +

# Create .venv and install the locked dependencies
venv-install:
    uv sync --locked
    @just mflux-version
